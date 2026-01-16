"""Structured logging for FraudGuard.

Provides consistent, configurable logging across all modules with:
- Console output with colored formatting
- Optional file output for training runs
- Structured context for ML metrics

Example:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started", epoch=1, lr=0.01)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog.stdlib import BoundLogger


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> None:
    """Configure global logging settings.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to file.
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> BoundLogger:
    """Get a structured logger for the given module.
    
    Args:
        name: Module name, typically __name__.
        
    Returns:
        Configured structlog logger.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Model training", epoch=10, loss=0.05)
    """
    return structlog.get_logger(name)


# Configure logging on module import
configure_logging()
