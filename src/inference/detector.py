"""A thin, deployment-focused wrapper around a trained detector.

Encapsulates the operational concerns that raw Ultralytics does not:
    * **per-class confidence thresholds** (smoke vs. fire behave differently),
    * a well-defined **alert state** (none / smoke / fire / fire_and_smoke),
    * consistent annotation using the project's class colors,
    * loading everything from ``configs/inference.yaml`` (incl. named operating
      modes: high_recall / balanced / high_precision).

This class is shared by ``predict_image`` and the Streamlit app so behavior is
identical everywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src.data.dfire import CLASS_NAMES, FIRE_ID, SMOKE_ID
from src.utils import get_logger, load_config, resolve_device
from src.visualization.plot_predictions import draw_boxes

logger = get_logger("detector")


class AlertState(str, Enum):
    """Operational alert derived from the set of confident detections."""

    NONE = "none"
    SMOKE = "smoke"
    FIRE = "fire"
    FIRE_AND_SMOKE = "fire_and_smoke"


@dataclass
class Detection:
    """A single confident detection in pixel coordinates."""

    cls: int
    name: str
    conf: float
    xyxy: tuple[int, int, int, int]

    def as_dict(self) -> dict:
        return {
            "cls": self.cls,
            "name": self.name,
            "conf": round(self.conf, 4),
            "xyxy": list(self.xyxy),
        }


class WildfireDetector:
    """Load a detector and run threshold-aware inference with alert logic."""

    def __init__(
        self,
        weights: str,
        conf: dict[str, float] | None = None,
        iou: float = 0.6,
        imgsz: int = 640,
        device: str | int | None = None,
        max_det: int = 300,
        agnostic_nms: bool = False,
    ) -> None:
        self.weights = weights
        self.conf = {"smoke": 0.25, "fire": 0.25}
        if conf:
            self.conf.update({k: float(v) for k, v in conf.items()})
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.device = resolve_device(device)
        #: Wall-clock duration of the most recent ``predict`` call, in ms.
        self.last_inference_ms: float | None = None

        from ultralytics import YOLO

        logger.info("Loading detector weights: %s", weights)
        self.model = YOLO(weights)
        self._check_class_names()

    def _check_class_names(self) -> None:
        """Warn loudly if the loaded weights disagree with the D-Fire classes."""
        model_names = getattr(self.model, "names", None) or {}
        expected = {cid: name for cid, name in CLASS_NAMES.items()}
        actual = {int(k): str(v) for k, v in dict(model_names).items()}
        if actual and actual != expected:
            logger.warning(
                "Model class names %s differ from the D-Fire convention %s. "
                "Alert logic and per-class thresholds assume 0=smoke, 1=fire.",
                actual,
                expected,
            )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        mode: str | None = None,
        weights: str | None = None,
    ) -> WildfireDetector:
        """Build a detector from ``configs/inference.yaml``.

        Args:
            config_path: path to the inference config.
            mode: optional named operating point (high_recall / balanced /
                high_precision). If omitted, uses the config's ``conf`` block.
            weights: optional override of the weights path.
        """
        cfg = load_config(config_path)
        conf = cfg.get("conf", {})
        if mode:
            modes = cfg.get("modes", {})
            if mode not in modes:
                raise ValueError(f"Unknown mode '{mode}'. Available: {list(modes)}")
            conf = modes[mode]
            logger.info("Using operating mode '%s' with thresholds %s", mode, conf)
        return cls(
            weights=weights or cfg["weights"],
            conf=conf,
            iou=float(cfg.get("iou", 0.6)),
            imgsz=int(cfg.get("imgsz", 640)),
            device=cfg.get("device"),
            max_det=int(cfg.get("max_det", 300)),
            agnostic_nms=bool(cfg.get("agnostic_nms", False)),
        )

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run inference on a single image.

        Args:
            source: an image file path, or an **RGB** ``np.ndarray`` (HWC).
                Ultralytics expects numpy arrays in BGR (OpenCV) order, so RGB
                arrays are converted internally — callers can safely pass the
                RGB images they already use for display.

        Detections are filtered with **per-class** thresholds. We query the
        model at the minimum class threshold, then apply each class's own
        cutoff.
        """
        if isinstance(source, np.ndarray):
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        else:
            source = str(source)

        min_conf = max(0.001, min(self.conf.values()))
        start = time.perf_counter()
        results = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            iou=self.iou,
            conf=min_conf,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0

        res = results[0]
        detections: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy().astype(float)
        xyxys = res.boxes.xyxy.cpu().numpy().astype(int)
        for cid, cf, box in zip(cls_ids, confs, xyxys, strict=True):
            name = CLASS_NAMES.get(int(cid), str(cid))
            if cf >= self.conf.get(name, 0.25):
                detections.append(Detection(int(cid), name, float(cf), tuple(int(v) for v in box)))
        return detections

    @staticmethod
    def alert_state(detections: list[Detection]) -> AlertState:
        """Collapse detections into an :class:`AlertState`."""
        has_smoke = any(d.cls == SMOKE_ID for d in detections)
        has_fire = any(d.cls == FIRE_ID for d in detections)
        if has_fire and has_smoke:
            return AlertState.FIRE_AND_SMOKE
        if has_fire:
            return AlertState.FIRE
        if has_smoke:
            return AlertState.SMOKE
        return AlertState.NONE

    def annotate(self, image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw confident detections on an RGB image."""
        boxes = [(d.cls, *d.xyxy, d.conf) for d in detections]
        return draw_boxes(image_rgb, boxes)
