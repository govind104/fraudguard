"""Training callbacks for early stopping, checkpointing, and logging.

Example:
    >>> callbacks = [EarlyStopping(patience=30), ModelCheckpoint(path="models/")]
    >>> trainer = FraudTrainer(callbacks=callbacks)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Callback:
    """Base callback class."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module) -> bool:
        """Called at end of each epoch. Return False to stop training."""
        return True

    def on_train_end(self, metrics: Dict[str, float], model: torch.nn.Module) -> None:
        """Called at end of training."""
        pass


class EarlyStopping(Callback):
    """Stop training when monitored metric stops improving.

    Args:
        monitor: Metric name to monitor (default: 'val_gmeans').
        patience: Epochs without improvement before stopping.
        mode: 'max' or 'min' for metric direction.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "val_gmeans",
        patience: int = 30,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module) -> bool:
        current = metrics.get(self.monitor, 0)

        improved = (self.mode == "max" and current > self.best + self.min_delta) or (
            self.mode == "min" and current < self.best - self.min_delta
        )

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}", best=self.best)
                return False
        return True


class ModelCheckpoint(Callback):
    """Save model when monitored metric improves.

    Args:
        path: Directory to save checkpoints.
        monitor: Metric name to monitor.
        mode: 'max' or 'min' for metric direction.
        save_best_only: Only save when metric improves.
    """

    def __init__(
        self,
        path: str = "models",
        monitor: str = "val_gmeans",
        mode: str = "max",
        save_best_only: bool = True,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = float("-inf") if mode == "max" else float("inf")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module) -> bool:
        current = metrics.get(self.monitor, 0)

        improved = (self.mode == "max" and current > self.best) or (
            self.mode == "min" and current < self.best
        )

        if not self.save_best_only or improved:
            self.best = current
            save_path = self.path / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                save_path,
            )

            # Also save as best
            if improved:
                best_path = self.path / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                    },
                    best_path,
                )
                logger.info("Saved best model", epoch=epoch, **{self.monitor: current})

        return True

    def on_train_end(self, metrics: Dict[str, float], model: torch.nn.Module) -> None:
        final_path = self.path / "fraudguard_final.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            },
            final_path,
        )
        logger.info(f"Saved final model to {final_path}")


class MetricsLogger(Callback):
    """Log metrics to file and track history.

    Args:
        log_dir: Directory for log files.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module) -> bool:
        record = {"epoch": epoch, **metrics}
        self.history.append(record)

        # Log to file
        log_path = self.log_dir / "training_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return True

    def on_train_end(self, metrics: Dict[str, float], model: torch.nn.Module) -> None:
        # Save full history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
