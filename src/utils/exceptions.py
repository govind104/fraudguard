"""Custom exception hierarchy for FraudGuard.

Provides specific exceptions for different failure modes to enable
targeted error handling and informative error messages.

Example:
    >>> from src.utils.exceptions import DataLoadingError
    >>> raise DataLoadingError("train_transaction.csv not found", path="/data/raw")
"""


class FraudGuardError(Exception):
    """Base exception for all FraudGuard errors."""

    def __init__(self, message: str, **context):
        """Initialize with message and optional context.

        Args:
            message: Error description.
            **context: Additional context as key-value pairs.
        """
        self.message = message
        self.context = context
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class DataLoadingError(FraudGuardError):
    """Raised when data loading fails.

    Examples:
        - CSV file not found
        - Invalid file format
        - Missing required columns
    """

    pass


class PreprocessingError(FraudGuardError):
    """Raised when preprocessing fails.

    Examples:
        - Feature engineering errors
        - PCA fitting failures
        - Scaler errors
    """

    pass


class GraphBuildingError(FraudGuardError):
    """Raised when graph construction fails.

    Examples:
        - FAISS index errors
        - Edge construction failures
        - Data leakage detected
    """

    pass


class ConfigurationError(FraudGuardError):
    """Raised when configuration is invalid.

    Examples:
        - Missing config file
        - Invalid parameter values
        - Path resolution failures
    """

    pass


class ModelError(FraudGuardError):
    """Raised when model operations fail.

    Examples:
        - Invalid architecture parameters
        - Training failures
        - Checkpoint loading errors
    """

    pass


class InferenceError(FraudGuardError):
    """Raised when inference fails.

    Examples:
        - Input validation errors
        - Model not loaded
        - Prediction failures
    """

    pass
