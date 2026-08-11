"""Path helpers that work identically on a local machine and on Kaggle.

The project deliberately avoids hard-coded absolute paths. Everything is
resolved relative to the repository root, and the *dataset* root is resolved
through :func:`resolve_data_root`, which understands both local checkouts and
Kaggle's read-only ``/kaggle/input`` mounts.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (src/utils/paths.py -> repo/).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Standard output locations (see the repo layout in README.md).
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
WEIGHTS_DIR: Path = OUTPUTS_DIR / "weights"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_dirs() -> dict[str, Path]:
    """Return the standard output directories, creating them on demand."""
    dirs = {
        "outputs": OUTPUTS_DIR,
        "weights": WEIGHTS_DIR,
        "figures": FIGURES_DIR,
        "metrics": METRICS_DIR,
        "predictions": PREDICTIONS_DIR,
        "reports": REPORTS_DIR,
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def _looks_like_dfire_root(root: Path) -> bool:
    """A directory is a valid D-Fire root if it has train/val image folders."""
    return (root / "train" / "images").is_dir() and (root / "val" / "images").is_dir()


def resolve_data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the dataset root directory.

    Resolution order:
        1. ``explicit`` argument (e.g. from a ``--data-root`` CLI flag).
        2. The ``DFIRE_ROOT`` environment variable.
        3. ``<repo>/data`` (the local development layout used in this repo).
        4. Common Kaggle mount points under ``/kaggle/input``.

    Raises:
        FileNotFoundError: if no candidate contains ``train/images`` and
            ``val/images``.
    """
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get("DFIRE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(PROJECT_ROOT / "data")

    # Kaggle: scan the mounted input datasets for a D-Fire-shaped folder.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in sorted(kaggle_input.iterdir()):
            candidates.append(child)
            # Some Kaggle datasets nest one level deeper.
            if child.is_dir():
                candidates.extend(sorted(p for p in child.iterdir() if p.is_dir()))

    for cand in candidates:
        if cand and cand.is_dir() and _looks_like_dfire_root(cand):
            return cand.resolve()

    searched = "\n  - ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate a D-Fire dataset root containing 'train/images' and "
        "'val/images'. Pass --data-root explicitly or set $DFIRE_ROOT.\n"
        f"Searched:\n  - {searched}"
    )


def split_dirs(data_root: str | os.PathLike, split: str) -> tuple[Path, Path]:
    """Return ``(images_dir, labels_dir)`` for a given split under ``data_root``."""
    root = Path(data_root)
    return root / split / "images", root / split / "labels"
