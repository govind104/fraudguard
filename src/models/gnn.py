"""Graph Neural Network for fraud detection.

Implements a 3-layer GCN with batch normalization and dropout,
designed for binary fraud classification on transaction graphs.

Example:
    >>> model = FraudGNN(in_channels=32, hidden_channels=128)
    >>> logits = model(x, edge_index)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.utils.config import ModelConfig, load_model_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudGNN(nn.Module):
    """3-layer Graph Convolutional Network for fraud classification.
    
    Architecture:
        Input -> GCN(hidden) -> BN -> ReLU -> Dropout
              -> GCN(64) -> BN -> ReLU -> Dropout
              -> GCN(2) -> LogSoftmax
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Hidden layer size (default 128).
        num_classes: Output classes (default 2: fraud/non-fraud).
        dropout: Dropout probability (default 0.4).
        use_batch_norm: Whether to use batch normalization.
        
    Example:
        >>> model = FraudGNN(in_channels=32)
        >>> logits = model(x, edge_index)
        >>> pred = logits.argmax(dim=1)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
        use_batch_norm: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize FraudGNN.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: First hidden layer size.
            num_classes: Number of output classes.
            dropout: Dropout probability.
            use_batch_norm: Enable batch normalization.
            config: Optional config to override defaults.
        """
        super().__init__()
        
        # Override with config if provided
        if config:
            hidden_channels = config.gnn["hidden_channels"]
            dropout = config.gnn["dropout"]
            use_batch_norm = config.gnn.get("batch_norm", True)
        
        self.dropout_p = dropout
        self.use_batch_norm = use_batch_norm
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.conv3 = GCNConv(64, num_classes)
        
        # Batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(64)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        logger.info(
            "FraudGNN initialized",
            in_channels=in_channels,
            hidden=hidden_channels,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            
        Returns:
            Log-softmax probabilities of shape (N, num_classes).
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3 (output)
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings from second-to-last layer.
        
        Useful for visualization and downstream tasks.
        
        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            
        Returns:
            Node embeddings of shape (N, 64).
        """
        x = self.conv1(x, edge_index)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        
        return x


def create_gnn_from_config(
    in_channels: int,
    config: Optional[ModelConfig] = None,
) -> FraudGNN:
    """Factory function to create FraudGNN from config.
    
    Args:
        in_channels: Input feature dimension.
        config: Optional config. Loads default if not provided.
        
    Returns:
        Configured FraudGNN model.
    """
    config = config or load_model_config()
    
    return FraudGNN(
        in_channels=in_channels,
        hidden_channels=config.gnn["hidden_channels"],
        dropout=config.gnn["dropout"],
        use_batch_norm=config.gnn.get("batch_norm", True),
    )
