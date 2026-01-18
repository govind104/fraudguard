"""Full training pipeline orchestration for FraudGuard.

Implements the complete AD-RL-GNN training loop:
1. AdaptiveMCD downsampling on training data
2. RL-driven MCES subgraph selection
3. GNN training with FocalLoss
4. Validation monitoring with early stopping

Example:
    >>> trainer = FraudTrainer()
    >>> trainer.fit(train_df, val_df, test_df)
    >>> metrics = trainer.evaluate(test_df)
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.data.preprocessor import FeaturePreprocessor
from src.data.graph_builder import GraphBuilder
from src.models import FraudGNN, FocalLoss, AdaptiveMCD, RLAgent, MCES, compute_class_weights, train_adaptive_mcd
from src.training.evaluator import Evaluator
from src.training.callbacks import Callback, EarlyStopping, ModelCheckpoint, MetricsLogger
from src.utils.config import load_model_config, load_data_config, ModelConfig, DataConfig
from src.utils.device_utils import get_device, set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudTrainer:
    """Full AD-RL-GNN training pipeline.
    
    Orchestrates:
    1. Data preprocessing and graph construction
    2. AdaptiveMCD training and downsampling
    3. RL agent training for subgraph selection
    4. MCES subgraph enhancement
    5. GNN training with FocalLoss
    6. Validation and early stopping
    
    Args:
        model_config: Model hyperparameters.
        data_config: Data settings.
        callbacks: List of training callbacks.
        device: Compute device.
        
    Example:
        >>> trainer = FraudTrainer(callbacks=[EarlyStopping(patience=30)])
        >>> trainer.fit(train_df, val_df, test_df)
        >>> metrics = trainer.evaluate()
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_config = model_config or load_model_config()
        self.data_config = data_config or load_data_config()
        self.callbacks = callbacks or []
        self.device = device or get_device()
        
        # Components (initialized during fit)
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.mcd: Optional[AdaptiveMCD] = None
        self.rl_agent: Optional[RLAgent] = None
        self.mces: Optional[MCES] = None
        self.model: Optional[FraudGNN] = None
        self.evaluator = Evaluator()
        
        # Training state
        self.data: Optional[Data] = None
        self.train_mask: Optional[torch.Tensor] = None
        self.val_mask: Optional[torch.Tensor] = None
        self.test_mask: Optional[torch.Tensor] = None
        self.history: List[Dict] = []
    
    def fit(
        self,
        train_df,
        val_df,
        test_df,
        max_epochs: int = 100,
        use_mcd: bool = True,
        use_rl: bool = True,
    ) -> Dict[str, float]:
        """Train the full pipeline.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            test_df: Test DataFrame.
            max_epochs: Maximum training epochs.
            use_mcd: Whether to use AdaptiveMCD downsampling.
            use_rl: Whether to use RL subgraph selection.
            
        Returns:
            Final evaluation metrics.
        """
        start_time = time.time()
        logger.info("Starting FraudTrainer.fit()", use_mcd=use_mcd, use_rl=use_rl)
        
        # Step 1: Preprocess features
        self._preprocess(train_df, val_df, test_df)
        
        # Step 2: Build graph (leak-free)
        self._build_graph()
        
        # Step 3: Prepare labels and masks
        self._prepare_labels(train_df, val_df, test_df)
        
        # Step 4: AdaptiveMCD (optional)
        if use_mcd:
            self._train_mcd()
        
        # Step 5: RL agent and MCES enhancement (optional)
        if use_rl:
            self._train_rl_and_enhance()
        
        # Step 6: Initialize GNN
        self._init_model()
        
        # Step 7: Training loop
        self._train_loop(max_epochs)
        
        train_time = time.time() - start_time
        logger.info(f"Training complete", time=f"{train_time:.1f}s", epochs=len(self.history))
        
        # Final evaluation
        final_metrics = self.evaluate()
        final_metrics["train_time"] = train_time
        
        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_end(final_metrics, self.model)
        
        return final_metrics
    
    def _preprocess(self, train_df, val_df, test_df):
        """Preprocess features."""
        logger.info("Preprocessing features...")
        self.preprocessor = FeaturePreprocessor(self.data_config, self.model_config)
        
        self.X_train = self.preprocessor.fit_transform(train_df).to(self.device)
        self.X_val = self.preprocessor.transform(val_df).to(self.device)
        self.X_test = self.preprocessor.transform(test_df).to(self.device)
        self.X_full = torch.cat([self.X_train, self.X_val, self.X_test])
        
        self.train_size = len(self.X_train)
        self.val_size = len(self.X_val)
        self.test_size = len(self.X_test)
        
        logger.info(f"Features: {self.X_full.shape}")
    
    def _build_graph(self):
        """Build leak-free graph."""
        logger.info("Building graph (training data only)...")
        self.graph_builder = GraphBuilder(self.model_config)
        
        train_edges = self.graph_builder.fit(self.X_train)
        self.edge_index = self.graph_builder.transform(
            torch.cat([self.X_val, self.X_test]),
            train_size=self.train_size,
        ).to(self.device)
        
        self.graph_builder.verify_no_leakage(self.edge_index, self.train_size)
        logger.info(f"Graph: {self.edge_index.shape[1]} edges")
    
    def _prepare_labels(self, train_df, val_df, test_df):
        """Prepare labels and masks."""
        self.train_labels = torch.tensor(
            train_df["isFraud"].values, dtype=torch.long, device=self.device
        )
        self.val_labels = torch.tensor(
            val_df["isFraud"].values, dtype=torch.long, device=self.device
        )
        self.test_labels = torch.tensor(
            test_df["isFraud"].values, dtype=torch.long, device=self.device
        )
        self.all_labels = torch.cat([self.train_labels, self.val_labels, self.test_labels])
        
        # Masks
        n = len(self.X_full)
        self.train_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.val_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.test_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        
        self.train_mask[:self.train_size] = True
        self.val_mask[self.train_size:self.train_size+self.val_size] = True
        self.test_mask[self.train_size+self.val_size:] = True
        
        # Fraud/non-fraud indices in training set
        self.fraud_nodes = torch.where(self.train_labels == 1)[0]
        self.majority_nodes = torch.where(self.train_labels == 0)[0]
        
        logger.info(f"Labels: {self.fraud_nodes.shape[0]} fraud, {self.majority_nodes.shape[0]} non-fraud")
    
    def _train_mcd(self):
        """Train and apply AdaptiveMCD."""
        logger.info("Training AdaptiveMCD...")
        fraud_ratio = len(self.fraud_nodes) / self.train_size
        
        self.mcd = AdaptiveMCD(
            input_dim=self.X_full.shape[1],
            fraud_ratio=fraud_ratio,
            config=self.model_config,
        ).to(self.device)
        
        # Train MCD
        n_epochs = self.model_config.adaptive_mcd.get("training_epochs", 10)
        lr = self.model_config.adaptive_mcd.get("learning_rate", 0.001)
        train_adaptive_mcd(self.mcd, self.X_train[self.majority_nodes], n_epochs, lr)
        
        # Downsample
        self.kept_majority = self.mcd.downsample(
            self.X_train[self.majority_nodes],
            self.majority_nodes,
        )
        
        # CRITICAL FIX: Update train_mask to exclude dropped nodes from loss calculation
        # This is what actually **activates** the class balancing benefit of MCD
        self.train_mask[:] = False  # Reset mask
        self.train_mask[self.fraud_nodes] = True  # Keep all fraud nodes
        self.train_mask[self.kept_majority] = True  # Keep only selected majority nodes
        
        logger.info(
            f"MCD downsampled: kept {len(self.kept_majority)}/{len(self.majority_nodes)} majority samples. "
            f"New Training Set Size: {self.train_mask.sum()}"
        )
    
    def _train_rl_and_enhance(self):
        """Train RL agent and enhance graph with MCES."""
        logger.info("Training RL agent for subgraph selection...")
        
        self.rl_agent = RLAgent(
            feat_dim=self.X_full.shape[1],
            config=self.model_config,
        ).to(self.device)
        
        # Get all fraud nodes (train + val + test for subgraph enhancement)
        all_fraud = torch.where(self.all_labels == 1)[0]
        
        # Train RL on training fraud nodes
        self.rl_agent.train(
            fraud_nodes=self.fraud_nodes,
            features=self.X_full,
            edge_index=self.edge_index,
            labels=self.all_labels,
            n_epochs=self.model_config.rl_agent.get("training_epochs", 100),
        )
        
        # Enhance graph using all fraud nodes
        logger.info("Enhancing graph with MCES subgraphs...")
        enhanced_edges = self.rl_agent.enhance_graph(
            self.edge_index,
            all_fraud,
            self.X_full,
        )
        
        # Merge with downsampled graph (if MCD was used)
        if hasattr(self, "kept_majority") and self.kept_majority is not None:
            combined = torch.cat([self.fraud_nodes, self.kept_majority])
            mask = (
                torch.isin(self.edge_index[0], combined) &
                torch.isin(self.edge_index[1], combined)
            )
            mcd_edges = self.edge_index[:, mask]
            self.edge_index = torch.cat([enhanced_edges, mcd_edges], dim=1).unique(dim=1)
        else:
            self.edge_index = enhanced_edges
        
        logger.info(f"Enhanced graph: {self.edge_index.shape[1]} edges")
    
    def _init_model(self):
        """Initialize GNN model and optimizer."""
        self.model = FraudGNN(
            in_channels=self.X_full.shape[1],
            config=self.model_config,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config.training["learning_rate"],
            weight_decay=self.model_config.training["weight_decay"],
        )
        
        # Verified Fix: CrossEntropyLoss with 15x weight
        # Replaces FocalLoss which caused collapse
        fraud_weight = 15.0
        weights = torch.tensor([1.0, fraud_weight], device=self.device)
        logger.info(f"Using CrossEntropyLoss with fraud_weight={fraud_weight}")
        
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    def _train_loop(self, max_epochs: int):
        """Main training loop with validation."""
        val_interval = self.model_config.training.get("val_check_interval", 5)
        
        for epoch in range(max_epochs):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            out = self.model(self.X_full, self.edge_index)
            loss = self.criterion(out[self.train_mask], self.all_labels[self.train_mask])
            
            loss.backward()
            
            # Gradient clipping
            if self.model_config.training.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.model_config.training["gradient_clip"],
                )
            
            self.optimizer.step()
            
            metrics = {"epoch": epoch, "train_loss": loss.item()}
            
            # Validation
            if epoch % val_interval == 0:
                val_metrics = self._validate()
                metrics.update(val_metrics)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}",
                        train_loss=f"{loss.item():.4f}",
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
    
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.X_full, self.edge_index)
            val_loss = self.criterion(out[self.val_mask], self.all_labels[self.val_mask])
            
            pred = out[self.val_mask].argmax(dim=1).cpu().numpy()
            true = self.all_labels[self.val_mask].cpu().numpy()
            
            metrics = self.evaluator.compute_metrics(true, pred)
        
        return {
            "val_loss": val_loss.item(),
            "val_specificity": metrics["specificity"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_gmeans": metrics["gmeans"],
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.X_full, self.edge_index)
            pred = out[self.test_mask].argmax(dim=1).cpu().numpy()
            true = self.all_labels[self.test_mask].cpu().numpy()
        
        metrics = self.evaluator.compute_metrics(true, pred)
        logger.info(
            "Test evaluation",
            specificity=f"{metrics['specificity']:.4f}",
            gmeans=f"{metrics['gmeans']:.4f}",
            f1=f"{metrics['f1']:.4f}",
        )
        return metrics
    
    def benchmark_latency(self, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference latency."""
        return self.evaluator.benchmark_latency(
            self.model, self.X_full, self.edge_index, n_runs
        )
    
    def save(self, path: str = "models/fraudguard_final.pt"):
        """Save model and artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "in_channels": self.X_full.shape[1],
            },
        }, path)
        
        # Save preprocessor artifacts
        self.preprocessor.save_artifacts()
        
        logger.info(f"Model saved to {path}")
