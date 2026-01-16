"""Utility modules for logging, device management, and exceptions.

This subpackage provides:
- get_logger: Structured logging with console and file output
- get_device: Automatic GPU/CPU device detection
- Custom exception hierarchy for error handling
"""

from src.utils.logger import get_logger
from src.utils.device_utils import get_device
from src.utils.exceptions import (
    FraudGuardError,
    DataLoadingError,
    PreprocessingError,
    GraphBuildingError,
    ConfigurationError,
)

__all__ = [
    "get_logger",
    "get_device",
    "FraudGuardError",
    "DataLoadingError",
    "PreprocessingError",
    "GraphBuildingError",
    "ConfigurationError",
]
