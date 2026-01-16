"""Training modules for FraudGuard.

This subpackage contains:
- FraudTrainer: Full pipeline orchestration
- Evaluator: Metrics computation and benchmarking
- Callbacks: EarlyStopping, ModelCheckpoint, MetricsLogger
"""

from src.training.trainer import FraudTrainer
from src.training.evaluator import Evaluator
from src.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
)

__all__ = [
    "FraudTrainer",
    "Evaluator",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricsLogger",
]
