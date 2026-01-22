"""Focal Loss for class-imbalanced learning.

Implements focal loss that down-weights well-classified examples,
focusing training on hard negatives. Particularly effective for
fraud detection with severe class imbalance.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

Example:
    >>> criterion = FocalLoss(alpha=0.5, gamma=4)
    >>> loss = criterion(logits, targets)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.
    
    Applies a modulating factor (1 - p_t)^gamma to cross-entropy loss,
    reducing the loss contribution from easy examples and focusing
    on hard, misclassified examples.
    
    Args:
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter. Higher values focus more on hard examples.
        weight: Optional class weights tensor.
        reduction: Reduction method ('mean', 'sum', 'none').
        
    Example:
        >>> criterion = FocalLoss(alpha=0.5, gamma=4)
        >>> logits = model(x, edge_index)  # (N, 2)
        >>> loss = criterion(logits, labels)  # (N,)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """Initialize focal loss.
        
        Args:
            alpha: Balance factor for positive class (default 0.25).
            gamma: Focusing parameter (default 2.0, paper uses 2).
            weight: Optional per-class weights.
            reduction: How to reduce batch loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Logits of shape (N, C) for C classes.
            targets: Ground truth labels of shape (N,).
            
        Returns:
            Scalar loss if reduction='mean'/'sum', else (N,) tensor.
        """
        # Standard cross-entropy (unreduced)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction="none", 
            weight=self.weight,
        )
        
        # Probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Focal modulation
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(
    labels: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute inverse frequency class weights.
    
    Args:
        labels: Ground truth labels tensor.
        device: Device for output tensor.
        
    Returns:
        Tensor of shape (num_classes,) with class weights.
        
    Example:
        >>> weights = compute_class_weights(train_labels)
        >>> criterion = FocalLoss(weight=weights)
    """
    unique, counts = torch.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)
    
    weights = n_samples / (n_classes * counts.float())
    
    if device:
        weights = weights.to(device)
    
    logger.info(
        "Computed class weights",
        weights=weights.tolist(),
        class_counts=counts.tolist(),
    )
    
    return weights
