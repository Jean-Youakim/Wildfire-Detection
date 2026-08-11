"""Train a YOLO detector (Ultralytics) on D-Fire.

Config-driven: all hyperparameters live in ``configs/train_*.yaml`` so runs are
reproducible and self-documenting. CLI flags override the config for quick
experiments (e.g. ``--epochs 5`` for a smoke test).

Examples:
    # Main model
    python -m src.training.train_yolo --config configs/train_yolo_main.yaml

    # Kaggle, explicit dataset mount + smaller batch
    python -m src.training.train_yolo --config configs/train_yolo_main.yaml \
        --data-root /kaggle/input/smoke-fire-detection-yolo --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.common import add_common_train_args, overrides_from_args, run_training


def train(config_path: str, overrides: dict) -> Path:
    """Run YOLO training; return the path to the run directory."""
    return run_training(
        config_path=config_path,
        overrides=overrides,
        default_model="yolov8n.pt",
        default_experiment="yolo_run",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on D-Fire.")
    parser.add_argument("--config", required=True, help="Path to a train_*.yaml config.")
    add_common_train_args(parser)
    args = parser.parse_args()

    train(args.config, overrides_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
