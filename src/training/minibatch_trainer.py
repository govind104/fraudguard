"""Mini-batch training with NeighborLoader for large graphs.

Uses PyG's NeighborLoader for memory-efficient training on graphs
too large to fit in memory (e.g., 91M edges).

Example:
    >>> trainer = MiniBatchTrainer()
    >>> trainer.fit(data, train_mask, val_mask)
"""

from typing import Optional, List, Dict
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.models import FraudGNN, FocalLoss, compute_class_weights
from src.training.evaluator import Evaluator
from src.training.callbacks import Callback, EarlyStopping, ModelCheckpoint, MetricsLogger
from src.utils.config import load_model_config, ModelConfig
from src.utils.device_utils import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MiniBatchTrainer:
    """Memory-efficient mini-batch GNN trainer.
    
    Uses NeighborLoader to sample subgraphs instead of loading
    the entire graph into memory.
    
    Args:
        batch_size: Number of target nodes per batch.
        num_neighbors: Neighbors to sample per layer.
        config: Model configuration.
        callbacks: Training callbacks.
        device: Compute device.
    """
    
    def __init__(
        self,
        batch_size: int = 4096,
        num_neighbors: List[int] = [25, 10],
        config: Optional[ModelConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.config = config or load_model_config()
        self.callbacks = callbacks or []
        self.device = device or get_device()
        
        self.model: Optional[FraudGNN] = None
        self.evaluator = Evaluator()
        self.history: List[Dict] = []
    
    def fit(
        self,
        data: Data,
        max_epochs: int = 100,
    ) -> Dict[str, float]:
        """Train on data with mini-batches.
        
        Args:
            data: PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.
            max_epochs: Maximum training epochs.
            
        Returns:
            Final metrics dictionary.
        """
        start_time = time.time()
        
        # Create loaders
        train_loader = NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
        )
        
        val_loader = NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=data.val_mask,
            shuffle=False,
        )
        
        logger.info(
            "Mini-batch training setup",
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            batch_size=self.batch_size,
        )
        
        # Initialize model
        self.model = FraudGNN(
            in_channels=data.x.shape[1],
            config=self.config,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training["learning_rate"],
            weight_decay=self.config.training["weight_decay"],
        )
        
        # Class weights from training labels
        train_labels = data.y[data.train_mask].to(self.device)
        weights = compute_class_weights(train_labels, self.device)
        criterion = FocalLoss(
            alpha=self.config.focal_loss["alpha"],
            gamma=self.config.focal_loss["gamma"],
            weight=weights,
        )
        
        # Training loop
        val_interval = self.config.training.get("val_check_interval", 5)
        
        for epoch in range(max_epochs):
            # Train
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index)
                # Only use "seed" nodes (first batch_size nodes)
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            metrics = {"epoch": epoch, "train_loss": avg_loss}
            
            # Validation
            if epoch % val_interval == 0:
                val_metrics = self._validate(val_loader)
                metrics.update(val_metrics)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}",
                        train_loss=f"{avg_loss:.4f}",
                        val_gmeans=f"{val_metrics['val_gmeans']:.4f}",
                    )
            
            self.history.append(metrics)
            
            # Callbacks
            should_stop = False
            for cb in self.callbacks:
                if not cb.on_epoch_end(epoch, metrics, self.model):
                    should_stop = True
            
            if should_stop:
                break
        
        train_time = time.time() - start_time
        logger.info("Mini-batch training complete", time=f"{train_time:.1f}s")
        
        # Final callbacks
        for cb in self.callbacks:
            cb.on_train_end(metrics, self.model)
        
        metrics["train_time"] = train_time
        return metrics
    
    def _validate(self, loader: NeighborLoader) -> Dict[str, float]:
        """Validate on loader."""
        self.model.eval()
        all_preds, all_true = [], []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)
                pred = out[:batch.batch_size].argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(batch.y[:batch.batch_size].cpu().numpy())
        
        metrics = self.evaluator.compute_metrics(
            np.array(all_true), np.array(all_preds)
        )
        
        return {
            f"val_{k}": v for k, v in metrics.items()
            if k in ["specificity", "recall", "f1", "gmeans"]
        }
    
    def evaluate(self, data: Data) -> Dict[str, float]:
        """Evaluate on test set."""
        test_loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],  # Full neighborhood for eval
            batch_size=self.batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
        )
        
        self.model.eval()
        all_preds, all_true = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)
                pred = out[:batch.batch_size].argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(batch.y[:batch.batch_size].cpu().numpy())
        
        return self.evaluator.compute_metrics(
            np.array(all_true), np.array(all_preds)
        )
    
    def benchmark_latency(self, data: Data, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark batch inference latency."""
        loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            batch_size=self.batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
        )
        
        self.model.eval()
        latencies = []
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_runs:
                    break
                batch = batch.to(self.device)
                start = time.perf_counter()
                _ = self.model(batch.x, batch.edge_index)
                latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
