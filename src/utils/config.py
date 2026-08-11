"""YAML config loading/saving with light validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file does not parse to a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must be a mapping, got {type(data).__name__}")
    return data


def save_config(config: dict[str, Any], path: str | os.PathLike) -> Path:
    """Write ``config`` to ``path`` as YAML and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return p
