"""Small, dependency-free logging helper for consistent console output."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "wildfire", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide configured logger.

    Idempotent: repeated calls do not add duplicate handlers (important in
    notebooks where cells run multiple times).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
        root = logging.getLogger("wildfire")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True

    logger = logging.getLogger(f"wildfire.{name}" if name != "wildfire" else "wildfire")
    logger.setLevel(level)
    return logger
