"""Device resolution shared by training, evaluation and inference."""

from __future__ import annotations


def resolve_device(device: str | int | None) -> str | int:
    """Resolve a device spec to something Ultralytics understands.

    ``None`` -> GPU 0 if CUDA is available, else ``'cpu'``.
    Anything else is passed through unchanged (e.g. ``'cpu'``, ``0``, ``'0,1'``).
    """
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"
