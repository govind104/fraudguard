"""Device utilities for GPU/CPU management.

Provides automatic device detection and memory monitoring for PyTorch operations.

Example:
    >>> from src.utils.device_utils import get_device, get_memory_info
    >>> device = get_device()
    >>> print(f"Using: {device}")
    Using: cuda
"""

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available compute device.

    Args:
        prefer_cuda: If True, use CUDA when available.

    Returns:
        torch.device for computation.

    Example:
        >>> device = get_device()
        >>> tensor = torch.randn(100, 100).to(device)
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using CUDA device",
            device_name=torch.cuda.get_device_name(0),
            memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_memory_info() -> dict:
    """Get current GPU memory usage.

    Returns:
        Dictionary with allocated and cached memory in GB.
        Returns empty dict if CUDA not available.

    Example:
        >>> info = get_memory_info()
        >>> print(f"Allocated: {info.get('allocated_gb', 0):.2f} GB")
    """
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "cached_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def clear_cuda_cache() -> None:
    """Clear CUDA memory cache.

    Useful for freeing GPU memory between training phases.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Random seeds set", seed=seed)
