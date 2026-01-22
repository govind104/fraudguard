"""Adaptive Majority Class Downsampling (AdaptiveMCD).

Implements learned downsampling for class-imbalanced graphs.
Uses a neural network to score majority class samples and
probabilistically drop less informative examples.

Example:
    >>> mcd = AdaptiveMCD(input_dim=32, fraud_ratio=0.035)
    >>> kept_indices = mcd.downsample(X_majority)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCD(nn.Module):
    """Base Majority Class Downsampling module.
    
    Learns to score majority class samples and probabilistically
    keeps samples based on their learned importance.
    
    Args:
        input_dim: Feature dimension.
        gamma: Base drop probability (higher = more aggressive).
        hidden_dim: Hidden layer size.
        
    Example:
        >>> mcd = MCD(input_dim=32, gamma=0.5)
        >>> kept_indices = mcd(X_majority)
    """
    
    def __init__(
        self,
        input_dim: int,
        gamma: float = 0.01,
        hidden_dim: int = 64,
    ):
        """Initialize MCD.
        
        Args:
            input_dim: Input feature dimension.
            gamma: Drop probability multiplier.
            hidden_dim: Size of hidden layer.
        """
        super().__init__()
        self.gamma = gamma
        
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, X_majority: torch.Tensor) -> torch.Tensor:
        """Compute indices to keep from majority class.
        
        Args:
            X_majority: Features of majority class samples (N, D).
            
        Returns:
            Indices of samples to keep.
        """
        scores = torch.sigmoid(self.selector(X_majority)).squeeze(-1)
        drop_probs = self.gamma * (1 - scores)
        keep_mask = torch.bernoulli(1 - drop_probs)
        return torch.where(keep_mask.bool())[0]
    
    def get_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Get importance scores for samples.
        
        Args:
            X: Input features.
            
        Returns:
            Scores in [0, 1] indicating sample importance.
        """
        return torch.sigmoid(self.selector(X)).squeeze(-1)


class AdaptiveMCD(MCD):
    """Adaptive Majority Class Downsampling.
    
    Extends MCD with fraud-ratio-aware gamma computation.
    Automatically adjusts downsampling aggressiveness based
    on the class imbalance ratio.
    
    Args:
        input_dim: Feature dimension.
        fraud_ratio: Ratio of fraud samples in training data.
        alpha: Aggressiveness factor (0-1).
        hidden_dim: Hidden layer size.
        
    Example:
        >>> mcd = AdaptiveMCD(input_dim=32, fraud_ratio=0.035, alpha=0.5)
        >>> mcd_optimizer = torch.optim.Adam(mcd.parameters(), lr=0.001)
        >>> # Train MCD
        >>> mcd.train_step(X_majority, mcd_optimizer)
        >>> # Downsample
        >>> kept = mcd.downsample(X_majority, majority_indices)
    """
    
    def __init__(
        self,
        input_dim: int,
        fraud_ratio: float,
        alpha: float = 0.5,
        hidden_dim: int = 64,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize AdaptiveMCD.
        
        Args:
            input_dim: Input feature dimension.
            fraud_ratio: Proportion of positive (fraud) class.
            alpha: How aggressively to downsample (0-1).
            hidden_dim: Hidden layer size.
            config: Optional config for hyperparameters.
        """
        # Compute adaptive gamma based on class imbalance
        if config:
            alpha = config.adaptive_mcd["alpha"]
            hidden_dim = config.adaptive_mcd["hidden_dim"]
        
        gamma = (1 - fraud_ratio) * alpha
        super().__init__(input_dim, gamma=gamma, hidden_dim=hidden_dim)
        
        self.alpha = alpha
        self.fraud_ratio = fraud_ratio
        
        logger.info(
            "AdaptiveMCD initialized",
            input_dim=input_dim,
            fraud_ratio=f"{fraud_ratio:.2%}",
            alpha=alpha,
            gamma=f"{gamma:.4f}",
        )
    
    def train_step(
        self,
        X_majority: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step for MCD selector.
        
        Trains the selector to output high scores (encouraging keeping).
        
        Args:
            X_majority: Majority class features.
            optimizer: Optimizer for selector parameters.
            
        Returns:
            Loss value for this step.
        """
        optimizer.zero_grad()
        scores = self.selector(X_majority).squeeze(-1)
        # Train to output ones (keep all) - the gamma factor handles dropping
        loss = F.binary_cross_entropy_with_logits(
            scores, torch.ones_like(scores)
        )
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def downsample(
        self,
        X_majority: torch.Tensor,
        majority_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Downsample majority class and return global indices.
        
        Args:
            X_majority: Features of majority class samples.
            majority_indices: Global indices of majority samples.
            
        Returns:
            Global indices of kept samples.
        """
        with torch.no_grad():
            local_kept = self(X_majority)
            return majority_indices[local_kept]


def train_adaptive_mcd(
    mcd: AdaptiveMCD,
    X_majority: torch.Tensor,
    n_epochs: int = 10,
    lr: float = 0.001,
) -> AdaptiveMCD:
    """Train AdaptiveMCD module.
    
    Args:
        mcd: AdaptiveMCD module to train.
        X_majority: Majority class features.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        
    Returns:
        Trained AdaptiveMCD module.
    """
    optimizer = torch.optim.Adam(mcd.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        loss = mcd.train_step(X_majority, optimizer)
        if (epoch + 1) % 5 == 0:
            logger.debug(f"MCD Epoch {epoch+1}: loss={loss:.4f}")
    
    logger.info("MCD training complete", epochs=n_epochs, final_loss=loss)
    return mcd
