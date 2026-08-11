"""End-to-end evaluation of a trained detector on a D-Fire split.

Produces, for the chosen split (default: test):
    1. **Detection metrics** via Ultralytics ``val()``:
       mAP50-95, mAP50, per-class AP, mean precision/recall.
    2. **Image-level alert metrics** (per-class ROC-AUC/PR-AUC/P/R/F1 and the
       4-way category confusion matrix) — the operational view.
    3. **Calibration** (ECE, reliability diagram, score histograms).
    4. **Threshold tuning**: best-F1 confidence per class on the eval split,
       plus the named operating points (high_recall / balanced / high_precision).

All numbers are written to ``outputs/metrics/eval_<split>.json`` and figures to
``outputs/figures/``.

Examples:
    python -m src.evaluation.evaluate_detector \
        --weights outputs/weights/yolo_main/weights/best.pt --split test

    # Tune thresholds on val, then evaluate test with them:
    python -m src.evaluation.evaluate_detector --weights ... --split val --tune-thresholds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.dfire import CLASS_NAMES, iter_split, parse_label_file
from src.evaluation.calibration import run_calibration
from src.evaluation.image_level_metrics import (
    category_confusion,
    compute_image_level_metrics,
)
from src.training.common import prepare_dataset_yaml
from src.utils import get_logger, load_config, resolve_data_root, resolve_device
from src.utils.paths import FIGURES_DIR, METRICS_DIR, ensure_dir, split_dirs

logger = get_logger("evaluate")


def run_val_metrics(model, data_yaml: Path, split: str, imgsz: int, iou: float, device) -> dict:
    """Run Ultralytics validation and extract detection metrics."""
    logger.info("Running detection validation on split '%s'...", split)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        iou=iou,
        conf=0.001,          # low conf for a full PR sweep (standard for mAP)
        plots=True,
        verbose=False,
        device=device,
    )
    box = metrics.box
    per_class_ap = {}
    # metrics.box.maps -> per-class mAP50-95, indexed by class id present in model.names
    try:
        maps = list(box.maps)
        names = model.names  # dict id->name
        for cid, name in names.items():
            if cid < len(maps):
                per_class_ap[name] = float(maps[cid])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not extract per-class AP: %s", exc)

    return {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "mean_precision": float(box.mp),
        "mean_recall": float(box.mr),
        "per_class_ap50_95": per_class_ap,
    }


def collect_image_scores(model, images_dir: Path, imgsz: int, iou: float, device) -> dict:
    """Run prediction over a split and collect per-image ground-truth presence
    and per-class max confidence scores.

    Returns a dict with numpy arrays keyed by class name plus bookkeeping lists.
    """
    logger.info("Collecting per-image scores from: %s", images_dir)
    name_to_id = {name: cid for cid, name in CLASS_NAMES.items()}

    stems: list[str] = []
    y_true = {name: [] for name in CLASS_NAMES.values()}
    y_score = {name: [] for name in CLASS_NAMES.values()}

    # Ground-truth presence per image.
    gt_present: dict[str, dict[str, bool]] = {}
    image_paths: list[Path] = []
    for img_path, lbl_path in iter_split(images_dir):
        boxes, _ = parse_label_file(lbl_path)
        present = {name: any(b.cls == cid for b in boxes) for name, cid in name_to_id.items()}
        gt_present[img_path.stem] = present
        image_paths.append(img_path)

    # Predict at very low conf so every plausible detection contributes a score.
    # NOTE: images are fed in small chunks. Passing the full list at once makes
    # Ultralytics (>=8.1) load *every* image into a single batch tensor, which
    # OOMs the GPU on splits with thousands of images.
    batch_size = 16
    pairs: list[tuple[Path, object]] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        try:
            results = model.predict(
                source=[str(p) for p in chunk],
                imgsz=imgsz,
                iou=iou,
                conf=0.001,
                device=device,
                verbose=False,
            )
            pairs.extend(zip(chunk, results, strict=True))
        except Exception:  # noqa: BLE001 - retry per image so one bad file can't sink the chunk
            for p in chunk:
                try:
                    res = model.predict(
                        source=str(p), imgsz=imgsz, iou=iou, conf=0.001,
                        device=device, verbose=False,
                    )[0]
                    pairs.append((p, res))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Prediction failed for %s: %s", p.name, exc)

    for img_path, res in pairs:
        stem = img_path.stem
        # Max confidence per class for this image (0 if none).
        max_conf = {name: 0.0 for name in CLASS_NAMES.values()}
        if res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy().astype(float)
            for cid, cf in zip(cls_ids, confs, strict=True):
                name = CLASS_NAMES.get(int(cid))
                if name and cf > max_conf[name]:
                    max_conf[name] = float(cf)

        stems.append(stem)
        for name in CLASS_NAMES.values():
            y_true[name].append(int(gt_present.get(stem, {}).get(name, False)))
            y_score[name].append(max_conf[name])

    return {
        "stems": stems,
        "y_true": {k: np.asarray(v) for k, v in y_true.items()},
        "y_score": {k: np.asarray(v) for k, v in y_score.items()},
    }


def tune_thresholds(y_true: dict, y_score: dict, grid: np.ndarray | None = None) -> dict:
    """Per-class best-F1 threshold search on image-level presence."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    from sklearn.metrics import f1_score

    best: dict[str, float] = {}
    for name in CLASS_NAMES.values():
        yt = np.asarray(y_true[name]).astype(int)
        ys = np.asarray(y_score[name]).astype(float)
        if yt.sum() == 0:
            best[name] = 0.25
            continue
        f1s = [(t, f1_score(yt, (ys >= t).astype(int), zero_division=0)) for t in grid]
        best[name] = float(max(f1s, key=lambda x: x[1])[0])
    return best


def evaluate(
    weights: str,
    split: str,
    data_root: str | None,
    imgsz: int,
    iou: float,
    device_arg,
    inference_config: str,
    do_tune: bool,
) -> dict:
    from ultralytics import YOLO

    device = resolve_device(device_arg)
    data_yaml = prepare_dataset_yaml(data_root)
    resolved_root = resolve_data_root(data_root)
    images_dir, _ = split_dirs(resolved_root, split)

    logger.info("Loading weights: %s", weights)
    model = YOLO(weights)

    detection = run_val_metrics(model, data_yaml, split, imgsz, iou, device)
    scores = collect_image_scores(model, images_dir, imgsz, iou, device)

    # Thresholds: tuned on this split (if requested) or taken from inference.yaml.
    inf_cfg = load_config(inference_config)
    default_thr = {
        name: float(inf_cfg.get("conf", {}).get(name, 0.25)) for name in CLASS_NAMES.values()
    }
    tuned = None
    if do_tune:
        if split == "test":
            logger.warning(
                "Tuning thresholds on the TEST split leaks test information into "
                "the operating point. Tune on 'val' and report the chosen "
                "thresholds on 'test' instead."
            )
        tuned = tune_thresholds(scores["y_true"], scores["y_score"])
    thresholds = tuned or default_thr

    img_metrics = compute_image_level_metrics(scores["y_true"], scores["y_score"], thresholds)
    confusion = category_confusion(scores["y_true"], scores["y_score"], thresholds)

    # Calibration on the overall alert score.
    names = list(CLASS_NAMES.values())
    true_any = np.maximum.reduce([scores["y_true"][n].astype(int) for n in names])
    score_any = np.maximum.reduce([scores["y_score"][n].astype(float) for n in names])
    calibration = run_calibration(
        true_any, score_any, prefix=f"{split}_alert", figures_dir=FIGURES_DIR
    )

    report = {
        "weights": weights,
        "split": split,
        "imgsz": imgsz,
        "iou": iou,
        "detection": detection,
        "image_level": img_metrics.as_dict(),
        "category_confusion": confusion,
        "calibration": {k: v for k, v in calibration.items() if k != "bins"},
        "thresholds_used": thresholds,
        "thresholds_tuned": tuned,
        "operating_points": inf_cfg.get("modes", {}),
    }

    ensure_dir(METRICS_DIR)
    out_path = METRICS_DIR / f"eval_{split}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "=== Detection: mAP50-95=%.4f | mAP50=%.4f ===",
        detection["map50_95"],
        detection["map50"],
    )
    for name, ap in detection["per_class_ap50_95"].items():
        logger.info("    AP50-95[%s] = %.4f", name, ap)
    logger.info(
        "=== Alert: PR-AUC=%.4f | recall=%.4f | precision=%.4f ===",
        img_metrics.alert["pr_auc"],
        img_metrics.alert["recall"],
        img_metrics.alert["precision"],
    )
    logger.info("Thresholds used: %s", thresholds)
    logger.info("Wrote evaluation report -> %s", out_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector on D-Fire.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Search per-class best-F1 thresholds on this split (use --split val).",
    )
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        split=args.split,
        data_root=args.data_root,
        imgsz=args.imgsz,
        iou=args.iou,
        device_arg=args.device,
        inference_config=args.inference_config,
        do_tune=args.tune_thresholds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
