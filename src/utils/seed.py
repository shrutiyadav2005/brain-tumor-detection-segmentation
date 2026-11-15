"""Seed and device configuration utilities for reproducible experiments."""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seeds across python, numpy, and torch for reproducibility.

    Args:
        seed: Random seed integer.
        deterministic: If True, configures CuDNN backends to be deterministic.

    Returns:
        The seed integer that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        # Apple Silicon MPS support
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass

    return seed


def get_device(preference: str = "auto"):
    """Detect and return the optimal computing device.

    Supports:
        - NVIDIA CUDA GPUs
        - Apple Silicon MPS (Metal Performance Shaders)
        - CPU fallback

    Args:
        preference: Device preference ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        torch.device object or string representation if torch is not yet installed.
    """
    try:
        import torch

        pref = preference.lower().strip()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif pref == "cpu":
            return torch.device("cpu")

        # Auto detection priority: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"
