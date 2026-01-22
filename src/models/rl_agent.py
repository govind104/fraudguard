"""Reinforcement Learning Agent for subgraph selection.

Implements a policy network that learns to select optimal
subgraph sampling strategies (RW, K-hop, K-ego) for each node.

Example:
    >>> agent = RLAgent(feat_dim=32)
    >>> agent.train(fraud_nodes, features, edge_index, labels)
    >>> actions = agent.select_actions(features[fraud_nodes])
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mces import MCES
from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubgraphPolicy(nn.Module):
    """Policy network for subgraph method selection.
    
    Outputs action probabilities for each of 3 subgraph methods:
    0: Random Walk
    1: K-hop Neighbors
    2: K-ego Neighbors
    
    Args:
        input_dim: Feature dimension.
        hidden_dim: Hidden layer size.
        num_actions: Number of actions (default 3).
        
    Example:
        >>> policy = SubgraphPolicy(input_dim=32)
        >>> probs, logits = policy(node_features)
        >>> actions = torch.multinomial(probs, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_actions: int = 3,
    ):
        """Initialize policy network.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer size.
            num_actions: Number of possible actions.
        """
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features of shape (N, D) or (D,).
            
        Returns:
            Tuple of (action_probs, logits).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class RLAgent:
    """Reinforcement Learning agent for subgraph selection.
    
    Uses policy gradient to learn which subgraph sampling method
    works best for different node types. Reward is based on
    the number of fraud nodes found in the sampled subgraph.
    
    Args:
        feat_dim: Feature dimension.
        hidden_dim: Policy network hidden size.
        lr: Learning rate.
        reward_scaling: Multiplier for rewards.
        
    Example:
        >>> agent = RLAgent(feat_dim=32)
        >>> agent.train(fraud_nodes, features, edge_index, labels, epochs=100)
        >>> actions = agent.get_best_actions(features[fraud_nodes])
    """
    
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 16,
        lr: float = 0.005,
        reward_scaling: float = 2.0,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize RL agent.
        
        Args:
            feat_dim: Input feature dimension.
            hidden_dim: Policy hidden layer size.
            lr: Learning rate.
            reward_scaling: Reward multiplier.
            config: Optional config for hyperparameters.
        """
        if config:
            hidden_dim = config.rl_agent["hidden_dim"]
            lr = config.rl_agent["learning_rate"]
            reward_scaling = config.rl_agent["reward_scaling"]
        
        self.policy = SubgraphPolicy(feat_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.mces = MCES(config=config)
        self.reward_scaling = reward_scaling
        
        # Training history
        self.losses: List[float] = []
        self.rewards: List[float] = []
        
        logger.info(
            "RLAgent initialized",
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            lr=lr,
        )
    
    def to(self, device: torch.device) -> "RLAgent":
        """Move agent to device.
        
        Args:
            device: Target device.
            
        Returns:
            Self for chaining.
        """
        self.policy = self.policy.to(device)
        return self
    
    def train(
        self,
        fraud_nodes: torch.Tensor,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 100,
        sample_frac: float = 0.2,
    ) -> None:
        """Train the RL agent using policy gradient.
        
        Args:
            fraud_nodes: Indices of fraud nodes.
            features: Full node features.
            edge_index: Graph edge index.
            labels: Node labels (0/1).
            n_epochs: Number of training epochs.
            sample_frac: Fraction of fraud nodes to sample per epoch.
        """
        device = features.device
        labels = labels.to(device)
        
        # Sample subset of fraud nodes for training
        n_sample = max(1, int(len(fraud_nodes) * sample_frac))
        
        # Precompute adjacency
        self.mces.precompute_adjacency(edge_index.to(device))
        
        for epoch in range(n_epochs):
            # Random subset of fraud nodes
            perm = torch.randperm(len(fraud_nodes))[:n_sample]
            subset = fraud_nodes[perm].to(device)
            
            # Get node features
            node_features = features[subset]
            
            # Get action probabilities
            probs, logits = self.policy(node_features)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            
            # Compute rewards
            rewards = []
            for node, action in zip(subset.tolist(), actions.tolist()):
                method_nodes = self.mces.execute_method(node, action)
                if method_nodes:
                    method_tensor = torch.tensor(method_nodes, device=device)
                    valid_nodes = method_tensor[method_tensor < len(labels)]
                    fraud_found = labels[valid_nodes].sum().float()
                else:
                    fraud_found = torch.tensor(0.0, device=device)
                
                reward = torch.log1p(fraud_found * self.reward_scaling)
                rewards.append(reward)
            
            rewards = torch.stack(rewards)
            
            # Policy gradient loss
            log_probs = torch.log(probs[range(len(subset)), actions] + 1e-8)
            loss = -(log_probs * rewards).mean()
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record history
            self.losses.append(loss.item())
            self.rewards.append(rewards.mean().item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"RL Epoch {epoch+1}",
                    loss=f"{loss.item():.4f}",
                    avg_reward=f"{rewards.mean().item():.4f}",
                )
        
        logger.info(
            "RL training complete",
            epochs=n_epochs,
            final_loss=self.losses[-1],
            final_reward=self.rewards[-1],
        )
    
    def get_best_actions(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Get best actions for nodes (greedy).
        
        Args:
            features: Node features.
            
        Returns:
            Tensor of action indices.
        """
        self.policy.eval()
        with torch.no_grad():
            probs, _ = self.policy(features)
            return probs.argmax(dim=-1)
    
    def sample_actions(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Sample actions from policy (stochastic).
        
        Args:
            features: Node features.
            
        Returns:
            Tensor of sampled action indices.
        """
        self.policy.eval()
        with torch.no_grad():
            probs, _ = self.policy(features)
            return torch.multinomial(probs, 1).squeeze(-1)
    
    def enhance_graph(
        self,
        edge_index: torch.Tensor,
        fraud_nodes: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Enhance graph using learned policy.
        
        Args:
            edge_index: Original edge index.
            fraud_nodes: Fraud node indices.
            features: Node features.
            
        Returns:
            Enhanced edge index.
        """
        actions = self.get_best_actions(features[fraud_nodes])
        return self.mces.enhance_subgraph(edge_index, fraud_nodes, actions)
