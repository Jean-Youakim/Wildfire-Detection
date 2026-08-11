"""Reproducibility helpers: seed every RNG we might touch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs.

    Args:
        seed: The seed value.
        deterministic: If True, request deterministic cuDNN algorithms. This can
            slightly reduce throughput but makes runs reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch not installed in this environment (e.g. lightweight analysis).
        pass
