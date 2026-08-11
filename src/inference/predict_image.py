"""Run wildfire detection on one image or a folder of images.

Outputs, per image:
    * an annotated ``.png`` (boxes + confidences),
    * a structured JSON record with the alert state and detections.

Examples:
    python -m src.inference.predict_image --source data/test/images/WEB11682.jpg
    python -m src.inference.predict_image --source data/test/images --mode high_recall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.data.dfire import IMAGE_EXTENSIONS
from src.inference.detector import WildfireDetector
from src.utils import get_logger
from src.utils.paths import PREDICTIONS_DIR, ensure_dir

logger = get_logger("predict.image")


def _gather(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return [x for x in sorted(p.iterdir()) if x.suffix.lower() in IMAGE_EXTENSIONS]
    raise FileNotFoundError(f"Source not found: {source}")


def run(source: str, config: str, mode: str | None, weights: str | None, out_dir: str) -> dict:
    detector = WildfireDetector.from_config(config, mode=mode, weights=weights)
    out_root = ensure_dir(out_dir)
    records = []

    for img_path in _gather(source):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("Unreadable: %s", img_path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Predict on the already-decoded frame (avoids reading the file twice).
        detections = detector.predict(rgb)
        alert = detector.alert_state(detections)

        annotated = detector.annotate(rgb, detections)
        out_img = out_root / f"{img_path.stem}_pred.png"
        cv2.imwrite(str(out_img), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        record = {
            "image": str(img_path),
            "alert": alert.value,
            "num_detections": len(detections),
            "detections": [d.as_dict() for d in detections],
            "annotated": str(out_img),
            "inference_ms": round(detector.last_inference_ms or 0.0, 1),
        }
        records.append(record)
        logger.info("%s -> ALERT=%s (%d detections)", img_path.name, alert.value, len(detections))

    out_json = out_root / "predictions.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info("Wrote %d records -> %s", len(records), out_json)
    return {"records": records, "json": str(out_json)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Wildfire detection on image(s).")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument(
        "--mode", default=None, choices=["high_recall", "balanced", "high_precision"]
    )
    parser.add_argument("--weights", default=None, help="Override weights path.")
    parser.add_argument("--out", default=str(PREDICTIONS_DIR / "images"))
    args = parser.parse_args()

    run(args.source, args.config, args.mode, args.weights, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
