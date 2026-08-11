"""Shared utilities: paths, logging, reproducibility, config loading, devices."""

from .config import load_config, save_config
from .device import resolve_device
from .logging import get_logger
from .paths import (
    PROJECT_ROOT,
    ensure_dir,
    get_output_dirs,
    resolve_data_root,
)
from .reproducibility import set_seed

__all__ = [
    "PROJECT_ROOT",
    "ensure_dir",
    "get_output_dirs",
    "resolve_data_root",
    "resolve_device",
    "get_logger",
    "set_seed",
    "load_config",
    "save_config",
]
