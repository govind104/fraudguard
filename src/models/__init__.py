"""Model modules for FraudGuard.

This subpackage contains:
- FocalLoss: Class-imbalanced loss function
- FraudGNN: 3-layer GCN classifier
- AdaptiveMCD: Adaptive Majority Class Downsampling
- MCES: Minor-node-centered Explored Subgraph sampling
- RLAgent: RL policy for subgraph method selection
"""

from src.models.losses import FocalLoss, compute_class_weights
from src.models.gnn import FraudGNN, create_gnn_from_config
from src.models.adaptive_mcd import MCD, AdaptiveMCD, train_adaptive_mcd
from src.models.mces import MCES
from src.models.rl_agent import SubgraphPolicy, RLAgent

__all__ = [
    "FocalLoss",
    "compute_class_weights",
    "FraudGNN",
    "create_gnn_from_config",
    "MCD",
    "AdaptiveMCD",
    "train_adaptive_mcd",
    "MCES",
    "SubgraphPolicy",
    "RLAgent",
]
