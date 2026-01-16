"""Training modules for FraudGuard.

This subpackage contains:
- FraudTrainer: Full pipeline orchestration
- MiniBatchTrainer: Memory-efficient mini-batch training with NeighborLoader
- Evaluator: Metrics computation and benchmarking
- Callbacks: EarlyStopping, ModelCheckpoint, MetricsLogger
"""

from src.training.trainer import FraudTrainer
from src.training.minibatch_trainer import MiniBatchTrainer
from src.training.evaluator import Evaluator
from src.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
)

__all__ = [
    "FraudTrainer",
    "MiniBatchTrainer",
    "Evaluator",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricsLogger",
]
