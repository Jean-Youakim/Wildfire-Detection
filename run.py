"""Single entry point for the whole project.

Commands:
    python run.py demo                 # open the web demo (easiest way to try it)
    python run.py detect photo.jpg     # check one image from the terminal
    python run.py train                # train the model (needs GPU + dataset images)
    python run.py evaluate             # measure model quality (needs trained weights)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "outputs" / "weights" / "yolo_main" / "weights" / "best.pt"
TRAIN_CONFIG = REPO_ROOT / "configs" / "train_yolo_main.yaml"


def cmd_demo() -> int:
    """Launch the simple Streamlit demo."""
    app = REPO_ROOT / "app" / "streamlit_app.py"
    print("Opening demo in your browser...")
    print("Upload a photo to see if fire or smoke is detected.")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def cmd_detect(image_path: str) -> int:
    """Detect fire/smoke in a single image and print a plain-English result."""
    import cv2

    from src.inference.detector import WildfireDetector

    path = Path(image_path)
    if not path.is_file():
        print(f"ERROR: Image not found: {path}")
        return 1

    weights = DEFAULT_WEIGHTS
    if not weights.is_file():
        print(f"ERROR: No trained model at {weights}")
        print("Train first:  python run.py train")
        print("Or point configs/inference.yaml at an existing .pt file.")
        return 1

    print("Loading model...")
    detector = WildfireDetector.from_config(DEFAULT_CONFIG)

    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"ERROR: Could not read image: {path}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"Analyzing: {path.name}")
    detections = detector.predict(rgb)
    alert = detector.alert_state(detections)

    print()
    print("=" * 50)
    print(f"RESULT: {alert.value.upper().replace('_', ' ')}")
    print("=" * 50)

    if detections:
        print(f"Found {len(detections)} detection(s):")
        for d in detections:
            print(f"  - {d.name}: {d.conf:.0%} confidence")
    else:
        print("No fire or smoke detected above the confidence threshold.")

    print(f"Inference time: {detector.last_inference_ms:.0f} ms")
    return 0


def cmd_train() -> int:
    """Train the main YOLO model."""
    if not TRAIN_CONFIG.is_file():
        print(f"ERROR: Training config not found: {TRAIN_CONFIG}")
        return 1

    print("Starting training (this takes a while — use a GPU if possible)...")
    print("Make sure your dataset images are in data/train/images, etc.")
    from src.training.train_yolo import train

    train(str(TRAIN_CONFIG), overrides={"data_root": None, "model": None, "device": None,
                                         "epochs": None, "batch": None, "imgsz": None})
    print()
    print("Done! Weights saved to outputs/weights/yolo_main/weights/best.pt")
    return 0


def cmd_evaluate() -> int:
    """Run evaluation on the test split."""
    if not DEFAULT_WEIGHTS.is_file():
        print(f"ERROR: No trained model at {DEFAULT_WEIGHTS}")
        print("Train first:  python run.py train")
        return 1

    print("Running evaluation on the test set...")
    from src.evaluation.evaluate_detector import evaluate

    evaluate(
        weights=str(DEFAULT_WEIGHTS),
        split="test",
        data_root=None,
        imgsz=640,
        iou=0.6,
        device_arg=None,
        inference_config=str(DEFAULT_CONFIG),
        do_tune=False,
    )
    print()
    print("Results saved to outputs/metrics/eval_test.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wildfire Detection — simple commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="Open the web demo (recommended)")
    detect_p = sub.add_parser("detect", help="Check one image from the terminal")
    detect_p.add_argument("image", help="Path to a .jpg or .png file")
    sub.add_parser("train", help="Train the model (needs GPU + dataset)")
    sub.add_parser("evaluate", help="Measure model quality on test set")

    args = parser.parse_args()
    if args.command == "demo":
        return cmd_demo()
    if args.command == "detect":
        return cmd_detect(args.image)
    if args.command == "train":
        return cmd_train()
    if args.command == "evaluate":
        return cmd_evaluate()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
