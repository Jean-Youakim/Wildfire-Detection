"""Shared helpers for the training/eval entry points.

Contains:
    * :func:`prepare_dataset_yaml` — turn the repo-relative
      ``configs/dataset.yaml`` into a fully resolved, absolute-path dataset
      YAML that Ultralytics can consume on any machine (local or Kaggle),
      without mutating the version-controlled file.
    * :func:`run_training` — the config-driven YOLO training flow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils import (
    get_logger,
    load_config,
    resolve_data_root,
    resolve_device,
    save_config,
    set_seed,
)
from src.utils.paths import METRICS_DIR, OUTPUTS_DIR, PROJECT_ROOT, WEIGHTS_DIR, ensure_dir

logger = get_logger("train.common")

DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"

# Hyperparameters forwarded verbatim to Ultralytics ``train()``. Anything else
# in the YAML (e.g. experiment_name, model) is handled by us, not passed
# through, so a stray key cannot silently change training behavior.
PASSTHROUGH_KEYS = frozenset({
    # core schedule
    "imgsz", "epochs", "batch", "patience", "optimizer", "lr0", "lrf", "cos_lr",
    "warmup_epochs", "weight_decay", "amp", "seed", "deterministic",
    # augmentation
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "fliplr", "flipud", "mosaic", "close_mosaic", "mixup",
    "erasing",
    # loss weights / gradient accumulation
    "nbs", "box", "cls", "dfl",
    # data loading & bookkeeping
    "workers", "cache", "save_period", "plots",
})


def prepare_dataset_yaml(
    data_root: str | None = None,
    dataset_config: str | Path = DEFAULT_DATASET_CONFIG,
) -> Path:
    """Write a resolved dataset YAML with an absolute ``path`` and return it.

    Ultralytics resolves ``train``/``val``/``test`` relative to ``path``. We set
    ``path`` to the auto-detected (or explicit) dataset root so training works
    identically whether the data lives in ``./data`` or ``/kaggle/input/...``.
    """
    cfg = load_config(dataset_config)
    root = resolve_data_root(data_root)

    resolved = dict(cfg)
    resolved["path"] = str(root)

    # Sanity-check that the referenced splits exist.
    for split_key in ("train", "val", "test"):
        rel = cfg.get(split_key)
        if rel and not (root / rel).exists():
            logger.warning("Split '%s' path does not exist: %s", split_key, root / rel)

    out = ensure_dir(OUTPUTS_DIR) / "dataset.resolved.yaml"
    save_config(resolved, out)
    logger.info("Resolved dataset YAML -> %s (path=%s)", out, root)
    return out


def run_training(
    config_path: str,
    overrides: dict[str, Any],
    default_model: str = "yolov8n.pt",
    default_experiment: str = "yolo_run",
) -> Path:
    """Config-driven YOLO training.

    Args:
        config_path: path to a ``configs/train_*.yaml`` file.
        overrides: CLI overrides; recognized hyperparameters replace the
            config values, plus the special keys ``data_root``, ``model`` and
            ``device``.
        default_model: checkpoint used if neither config nor overrides set one.
        default_experiment: run name used if the config does not set one.

    Returns:
        The Ultralytics run directory (contains ``weights/best.pt``).
    """
    cfg = load_config(config_path)
    for key, value in overrides.items():
        if value is not None and key in PASSTHROUGH_KEYS:
            cfg[key] = value

    experiment = cfg.get("experiment_name", default_experiment)
    model_name = overrides.get("model") or cfg.get("model", default_model)
    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))

    data_yaml = prepare_dataset_yaml(overrides.get("data_root"))
    device = resolve_device(overrides.get("device"))

    train_kwargs = {k: cfg[k] for k in cfg if k in PASSTHROUGH_KEYS}
    train_kwargs.update(
        data=str(data_yaml),
        device=device,
        project=str(ensure_dir(WEIGHTS_DIR)),
        name=experiment,
        exist_ok=True,
        verbose=True,
    )

    logger.info("Loading YOLO model: %s", model_name)
    from ultralytics import YOLO

    model = YOLO(model_name)

    logger.info("Starting training '%s' on device=%s", experiment, device)
    results = model.train(**train_kwargs)

    run_dir = Path(getattr(results, "save_dir", WEIGHTS_DIR / experiment))
    logger.info("Training finished. Run dir: %s", run_dir)

    # Persist a compact summary next to our other metrics.
    summary = {
        "experiment": experiment,
        "model": model_name,
        "device": str(device),
        "data": str(data_yaml),
        "run_dir": str(run_dir),
        "best_weights": str(run_dir / "weights" / "best.pt"),
        "config": cfg,
    }
    ensure_dir(METRICS_DIR)
    summary_path = METRICS_DIR / f"train_{experiment}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote training summary -> %s", summary_path)
    return run_dir


def add_common_train_args(parser) -> None:
    """Register the CLI flags shared by every training entry point."""
    parser.add_argument(
        "--data-root", default=None, help="Dataset root (auto-detected if omitted)."
    )
    parser.add_argument("--model", default=None, help="Override the model checkpoint.")
    parser.add_argument(
        "--device", default=None, help="cuda index, 'cpu', or '0,1'. Auto if omitted."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs (handy for smoke tests)."
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")


def overrides_from_args(args) -> dict[str, Any]:
    """Map parsed CLI args to the overrides dict consumed by :func:`run_training`."""
    return {
        "data_root": args.data_root,
        "model": args.model,
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
    }
