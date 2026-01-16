"""Minor-node-centered Explored Subgraph (MCES) sampling.

Implements three subgraph sampling strategies centered on fraud nodes:
1. Random Walk (RW): Stochastic graph exploration
2. K-hop Neighbors: Breadth-first expansion
3. K-ego Neighbors: Ego-network extraction

Example:
    >>> mces = MCES(k_rw=10, k_hop=3, k_ego=2)
    >>> mces.precompute_adjacency(edge_index)
    >>> neighbors = mces.random_walk(node_id)
"""

import torch
from typing import List, Optional, Tuple

from src.utils.config import load_model_config, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCES:
    """Minor-node-centered Explored Subgraph sampler.
    
    Provides three sampling strategies for extracting subgraphs
    centered on minority (fraud) nodes:
    
    1. Random Walk: Explores graph stochastically
    2. K-hop: All nodes within K edges
    3. K-ego: Ego-network to depth K
    
    Precomputes adjacency for efficient repeated sampling.
    
    Args:
        k_rw: Random walk length.
        k_hop: K-hop neighborhood depth.
        k_ego: K-ego network depth.
        max_neighbors: Max neighbors to store per node.
        
    Example:
        >>> mces = MCES()
        >>> mces.precompute_adjacency(edge_index)
        >>> for node in fraud_nodes:
        ...     neighbors = mces.execute_method(node, action=0)
    """
    
    def __init__(
        self,
        k_rw: int = 10,
        k_hop: int = 3,
        k_ego: int = 2,
        max_neighbors: int = 1000,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize MCES sampler.
        
        Args:
            k_rw: Random walk length (default 10).
            k_hop: K-hop depth (default 3).
            k_ego: K-ego depth (default 2).
            max_neighbors: Max neighbors per node.
            config: Optional config for hyperparameters.
        """
        if config:
            k_rw = config.mces["k_rw"]
            k_hop = config.mces["k_hop"]
            k_ego = config.mces["k_ego"]
            max_neighbors = config.mces["max_neighbors"]
        
        self.k_rw = k_rw
        self.k_hop = k_hop
        self.k_ego = k_ego
        self.max_neighbors = max_neighbors
        
        # Precomputed adjacency matrix (node -> neighbors)
        self.adj_matrix: Optional[torch.Tensor] = None
        self.num_nodes: int = 0
        
        logger.info(
            "MCES initialized",
            k_rw=k_rw,
            k_hop=k_hop,
            k_ego=k_ego,
        )
    
    def precompute_adjacency(self, edge_index: torch.Tensor) -> None:
        """Build adjacency matrix for efficient neighbor lookup.
        
        Args:
            edge_index: Edge index tensor of shape (2, E).
        """
        self.num_nodes = edge_index.max().item() + 1
        device = edge_index.device
        
        # Initialize with -1 (no neighbor)
        self.adj_matrix = torch.full(
            (self.num_nodes, self.max_neighbors),
            -1,
            dtype=torch.long,
            device=device,
        )
        
        # Populate adjacency
        nodes, counts = torch.unique(edge_index[0], return_counts=True)
        for node, count in zip(nodes.tolist(), counts.tolist()):
            neighbors = edge_index[1][edge_index[0] == node][:self.max_neighbors]
            self.adj_matrix[node, :len(neighbors)] = neighbors
        
        logger.info(
            "Adjacency precomputed",
            num_nodes=self.num_nodes,
            avg_degree=edge_index.shape[1] / self.num_nodes,
        )
    
    def execute_method(
        self,
        node: int,
        action: int,
    ) -> List[int]:
        """Execute sampling method based on action.
        
        Args:
            node: Center node for sampling.
            action: 0=RW, 1=K-hop, 2=K-ego.
            
        Returns:
            List of neighbor node indices.
        """
        if action == 0:
            return self.random_walk(node)
        elif action == 1:
            return self.k_hop_neighbors(node)
        elif action == 2:
            return self.k_ego_neighbors(node)
        return []
    
    def random_walk(self, start_node: int) -> List[int]:
        """Perform random walk from start node.
        
        Args:
            start_node: Starting node for walk.
            
        Returns:
            List of visited node indices.
        """
        assert self.adj_matrix is not None, "Call precompute_adjacency first"
        device = self.adj_matrix.device
        
        walk = torch.full((self.k_rw,), -1, dtype=torch.long, device=device)
        current = torch.tensor(start_node, device=device, dtype=torch.long)
        
        for step in range(self.k_rw):
            neighbors = self.adj_matrix[current]
            valid = neighbors[neighbors >= 0]
            
            if len(valid) == 0:
                break
            
            # Random neighbor selection
            idx = torch.randint(0, len(valid), (1,), device=device)
            current = valid[idx]
            walk[step] = current
        
        return walk[walk >= 0].tolist()
    
    def k_hop_neighbors(self, start_node: int) -> List[int]:
        """Get all nodes within K hops.
        
        Args:
            start_node: Center node.
            
        Returns:
            List of neighbor node indices.
        """
        assert self.adj_matrix is not None, "Call precompute_adjacency first"
        device = self.adj_matrix.device
        
        current_nodes = torch.tensor([start_node], device=device)
        visited = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        visited[start_node] = True
        
        for _ in range(self.k_hop):
            # Get all neighbors of current frontier
            neighbors = self.adj_matrix[current_nodes].flatten()
            neighbors = neighbors[neighbors >= 0]
            
            # Filter to unvisited
            new_nodes = neighbors[~visited[neighbors]]
            visited[new_nodes] = True
            current_nodes = new_nodes
        
        return visited.nonzero().flatten().tolist()
    
    def k_ego_neighbors(self, start_node: int) -> List[int]:
        """Get K-ego network (BFS to depth K).
        
        Args:
            start_node: Center node.
            
        Returns:
            List of neighbor node indices.
        """
        assert self.adj_matrix is not None, "Call precompute_adjacency first"
        device = self.adj_matrix.device
        
        visited = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        queue = torch.tensor([start_node], device=device)
        visited[start_node] = True
        
        for _ in range(self.k_ego):
            neighbors = self.adj_matrix[queue].flatten()
            neighbors = neighbors[neighbors >= 0]
            
            new_nodes = neighbors[~visited[neighbors]]
            visited[new_nodes] = True
            queue = new_nodes
        
        return visited.nonzero().flatten().tolist()
    
    def enhance_subgraph(
        self,
        edge_index: torch.Tensor,
        fraud_nodes: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enhance graph with subgraph edges around fraud nodes.
        
        Args:
            edge_index: Original edge index.
            fraud_nodes: Indices of fraud nodes.
            actions: Per-node actions (0/1/2). If None, uses all methods.
            
        Returns:
            Enhanced edge index with additional subgraph edges.
        """
        if self.adj_matrix is None:
            self.precompute_adjacency(edge_index)
        
        device = edge_index.device
        sub_edges = []
        
        for i, node in enumerate(fraud_nodes.tolist()):
            if actions is not None:
                action = actions[i].item()
                neighbors = self.execute_method(node, action)
                self._add_edges(sub_edges, node, neighbors, device)
            else:
                # Use all methods
                for action in [0, 1, 2]:
                    neighbors = self.execute_method(node, action)
                    self._add_edges(sub_edges, node, neighbors, device)
        
        if not sub_edges:
            return edge_index
        
        # Combine with original edges
        sub_tensors = [e for e in sub_edges if e.shape[0] == 2]
        if not sub_tensors:
            return edge_index
        
        combined = torch.cat([edge_index] + sub_tensors, dim=1)
        return combined.unique(dim=1)
    
    def _add_edges(
        self,
        sub_edges: List[torch.Tensor],
        node: int,
        neighbors: List[int],
        device: torch.device,
    ) -> None:
        """Helper to add bidirectional edges."""
        if not neighbors:
            return
        
        node_t = torch.tensor([node], device=device)
        neighbors_t = torch.tensor(neighbors, device=device)
        
        src = torch.cat([node_t.expand(len(neighbors)), neighbors_t])
        dst = torch.cat([neighbors_t, node_t.expand(len(neighbors))])
        
        valid = (dst >= 0) & (src >= 0)
        edges = torch.stack([src[valid], dst[valid]], dim=0)
        sub_edges.append(edges)
