"""Leak-free FAISS-based semantic similarity graph builder.

Provides the GraphBuilder class for constructing graph edges using
cosine similarity, with strict train/test separation to prevent data leakage.

CRITICAL: This module ensures graph edges are built using ONLY training data.
Test nodes can be added later but will only connect to training nodes.

Supports both faiss-cpu and faiss-gpu with automatic detection.

Example:
    >>> from src.data.graph_builder import GraphBuilder
    >>> builder = GraphBuilder()
    >>> # Build graph using ONLY training data
    >>> train_edges = builder.fit(X_train)
    >>> # Add test nodes (connects to train only, no test-test edges)
    >>> full_edges = builder.transform(X_test, train_size=len(X_train))
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# Try to import faiss-gpu, fallback to faiss-cpu
try:
    import faiss
    # Check if GPU is available
    if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
        FAISS_GPU_AVAILABLE = True
        print("Loading faiss with GPU support.")
    else:
        FAISS_GPU_AVAILABLE = False
        print("Loading faiss with CPU support (no GPU detected).")
except ImportError:
    import faiss
    FAISS_GPU_AVAILABLE = False
    print("Loading faiss-cpu.")

from src.utils.config import load_model_config, ModelConfig
from src.utils.exceptions import GraphBuildingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """FAISS-based semantic similarity graph builder with leak prevention.
    
    Constructs graph edges based on cosine similarity between node features.
    Uses FAISS for efficient k-NN search with configurable threshold.
    
    CRITICAL: The fit() method builds edges using ONLY training data to
    prevent data leakage. The transform() method adds test nodes that
    connect only to training nodes (no test-test edges).
    
    Attributes:
        threshold: Similarity threshold for edge creation.
        batch_size: Batch size for FAISS search.
        max_neighbors: Maximum neighbors to search per node.
        
    Example:
        >>> builder = GraphBuilder()
        >>> train_edges = builder.fit(X_train)
        >>> full_edges = builder.transform(X_test, train_size=len(X_train))
        >>> # Verify: no test-test edges exist
        >>> builder.verify_no_leakage(full_edges, train_size=len(X_train))
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize graph builder.
        
        Args:
            config: Optional ModelConfig. Loads from default if not provided.
        """
        config = config or load_model_config()
        
        self.threshold = config.graph.similarity_threshold
        self.batch_size = config.graph.batch_size
        self.max_neighbors = config.graph.max_neighbors
        
        # Store fitted FAISS index for transform
        self._index: Optional[faiss.Index] = None
        self._train_size: Optional[int] = None
        self._train_edges: Optional[torch.Tensor] = None
    
    def fit(self, X_train: torch.Tensor) -> torch.Tensor:
        """Build edge index using ONLY training data.
        
        Creates a FAISS index from training features and constructs
        edges between training nodes that exceed the similarity threshold.
        
        Args:
            X_train: Training feature tensor of shape (n_train, n_features).
            
        Returns:
            Edge index tensor of shape (2, n_edges) with training edges only.
            
        Raises:
            GraphBuildingError: If FAISS index construction fails.
            
        Example:
            >>> builder = GraphBuilder()
            >>> train_edges = builder.fit(X_train)
            >>> print(f"Train edges: {train_edges.shape[1]}")
        """
        logger.info(
            "Building FAISS index (TRAINING ONLY)",
            num_nodes=X_train.shape[0],
            features=X_train.shape[1],
            threshold=self.threshold,
        )
        
        self._train_size = X_train.shape[0]
        
        try:
            # Normalize features for cosine similarity
            features_np = X_train.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(features_np)
            
            # Build FAISS index (GPU if available)
            index_flat = faiss.IndexFlatIP(features_np.shape[1])
            
            if FAISS_GPU_AVAILABLE:
                # Use GPU for faster indexing
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                logger.info("Using FAISS GPU index")
            else:
                self._index = index_flat
            
            self._index.add(features_np)
            
            # Construct edges from training data only
            self._train_edges = self._build_edges(features_np, start_idx=0)
            
            logger.info(
                "Training graph built (LEAK-FREE)",
                num_edges=self._train_edges.shape[1],
                avg_degree=self._train_edges.shape[1] / self._train_size,
            )
            
            return self._train_edges
            
        except Exception as e:
            raise GraphBuildingError(
                f"FAISS index construction failed: {str(e)}",
                stage="fit",
            )
    
    def transform(
        self,
        X_test: torch.Tensor,
        train_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Add test nodes connecting ONLY to training nodes.
        
        Creates edges from test nodes to their nearest training neighbors.
        CRITICAL: Does NOT create test-test edges to prevent leakage.
        
        Args:
            X_test: Test feature tensor of shape (n_test, n_features).
            train_size: Number of training nodes. Uses stored value if None.
            
        Returns:
            Combined edge index with train-train and test-train edges.
            No test-test edges are created.
            
        Raises:
            GraphBuildingError: If not fitted or transform fails.
        """
        if self._index is None or self._train_edges is None:
            raise GraphBuildingError(
                "GraphBuilder not fitted. Call fit() first.",
                stage="transform",
            )
        
        train_size = train_size or self._train_size
        test_size = X_test.shape[0]
        
        logger.info(
            "Adding test nodes (NO test-test edges)",
            test_nodes=test_size,
            train_nodes=train_size,
        )
        
        try:
            # Normalize test features
            test_np = X_test.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(test_np)
            
            # Find nearest TRAINING neighbors for each test node
            edge_list = []
            
            for i in range(0, test_size, self.batch_size):
                batch = test_np[i:i+self.batch_size]
                similarities, neighbors = self._index.search(batch, self.max_neighbors)
                
                for idx_in_batch, (sim_row, nbr_row) in enumerate(zip(similarities, neighbors)):
                    test_idx = train_size + i + idx_in_batch
                    
                    # Only connect to training nodes above threshold
                    valid = sim_row > self.threshold
                    for train_idx in nbr_row[valid]:
                        if train_idx >= 0 and train_idx < train_size:
                            # Bidirectional edges
                            edge_list.append([test_idx, train_idx])
                            edge_list.append([train_idx, test_idx])
            
            if edge_list:
                test_edges = torch.tensor(edge_list, dtype=torch.long).t()
                combined = torch.cat([self._train_edges, test_edges], dim=1)
                combined = combined.unique(dim=1)
            else:
                combined = self._train_edges
            
            logger.info(
                "Test nodes added",
                test_train_edges=len(edge_list) // 2,
                total_edges=combined.shape[1],
            )
            
            return combined
            
        except Exception as e:
            raise GraphBuildingError(
                f"Transform failed: {str(e)}",
                stage="transform",
            )
    
    def _build_edges(
        self,
        features_np: np.ndarray,
        start_idx: int = 0,
    ) -> torch.Tensor:
        """Build edges from feature matrix using memory-optimized chunking.
        
        Uses vectorized numpy operations instead of Python loops for
        ~10x memory efficiency on large graphs.
        
        Args:
            features_np: Normalized feature matrix.
            start_idx: Starting index for node IDs.
            
        Returns:
            Edge index tensor (directed, not symmetrized to save memory).
        """
        import gc
        
        edge_chunks = []
        num_nodes = features_np.shape[0]
        
        logger.info(
            "Building edges (memory-optimized)",
            num_nodes=num_nodes,
            batch_size=self.batch_size,
            threshold=self.threshold,
        )
        
        for batch_start in range(0, num_nodes, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_nodes)
            batch = features_np[batch_start:batch_end]
            
            # FAISS search
            similarities, neighbors = self._index.search(batch, self.max_neighbors)
            
            # Vectorized filtering (no Python loops!)
            src_indices = np.arange(batch_start, batch_end) + start_idx
            src_expanded = src_indices[:, None]  # Shape: (batch, 1)
            
            # Broadcast source to match neighbor shape
            src_broadcast = np.broadcast_to(src_expanded, neighbors.shape)
            
            # Create mask: similarity > threshold, not self-loop, not invalid (-1)
            mask = (similarities > self.threshold) & (neighbors != src_expanded) & (neighbors >= 0)
            
            # Extract valid edges
            valid_src = src_broadcast[mask]
            valid_dst = neighbors[mask]
            
            if len(valid_src) > 0:
                # Convert to tensor immediately (compact memory)
                chunk_edges = torch.stack([
                    torch.from_numpy(valid_src.copy()),
                    torch.from_numpy(valid_dst.copy())
                ], dim=0).to(torch.long)
                edge_chunks.append(chunk_edges)
            
            # Aggressive cleanup
            del similarities, neighbors, mask, src_broadcast
            gc.collect()
        
        if not edge_chunks:
            logger.warning("No edges created. Consider lowering threshold.")
            return torch.empty((2, 0), dtype=torch.long)
        
        # Concatenate all chunks
        edge_tensor = torch.cat(edge_chunks, dim=1)
        
        # Sort by source node (helps NeighborLoader)
        # NOTE: We skip symmetrization to save 50% memory
        # Directed graph is valid for fraud detection KNN
        idx = edge_tensor[0].argsort()
        edge_tensor = edge_tensor[:, idx]
        
        logger.info(f"Built {edge_tensor.shape[1]:,} edges (directed)")
        return edge_tensor
    
    def verify_no_leakage(
        self,
        edge_index: torch.Tensor,
        train_size: int,
    ) -> bool:
        """Verify that no test-test edges exist (leak-free).
        
        CRITICAL: Call this method to validate graph construction.
        
        Args:
            edge_index: Edge index tensor of shape (2, n_edges).
            train_size: Number of training nodes.
            
        Returns:
            True if no test-test edges found.
            
        Raises:
            GraphBuildingError: If test-test edges are detected.
            
        Example:
            >>> builder.verify_no_leakage(full_edges, train_size=len(X_train))
            True
        """
        src = edge_index[0]
        dst = edge_index[1]
        
        # Find edges where BOTH nodes are test nodes
        test_test_mask = (src >= train_size) & (dst >= train_size)
        n_leaky_edges = test_test_mask.sum().item()
        
        if n_leaky_edges > 0:
            raise GraphBuildingError(
                f"DATA LEAKAGE DETECTED: {n_leaky_edges} test-test edges found!",
                test_test_edges=n_leaky_edges,
                train_size=train_size,
            )
        
        # Count edge types
        train_train = ((src < train_size) & (dst < train_size)).sum().item()
        train_test = ((src < train_size) != (dst < train_size)).sum().item()
        
        logger.info(
            "Graph verification PASSED (leak-free)",
            train_train_edges=train_train,
            train_test_edges=train_test // 2,  # Bidirectional
            test_test_edges=0,
        )
        
        return True
    
    def save_edges(
        self,
        edge_index: torch.Tensor,
        path: Optional[Path] = None,
        filename: str = "edges_faiss.pt",
    ) -> None:
        """Save edge index to disk.
        
        Args:
            edge_index: Edge index tensor to save.
            path: Directory path. Defaults to config graphs_dir.
            filename: Output filename.
        """
        from src.utils.config import load_data_config
        
        if path is None:
            path = load_data_config().paths.graphs_dir
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        save_path = path / filename
        torch.save(edge_index, save_path)
        
        logger.info("Edges saved", path=str(save_path), edges=edge_index.shape[1])
    
    def load_edges(
        self,
        path: Optional[Path] = None,
        filename: str = "edges_faiss.pt",
    ) -> torch.Tensor:
        """Load edge index from disk.
        
        Args:
            path: Directory path. Defaults to config graphs_dir.
            filename: Input filename.
            
        Returns:
            Loaded edge index tensor.
        """
        from src.utils.config import load_data_config
        
        if path is None:
            path = load_data_config().paths.graphs_dir
        
        path = Path(path) / filename
        
        if not path.exists():
            raise GraphBuildingError("Edge file not found", path=str(path))
        
        edge_index = torch.load(path)
        logger.info("Edges loaded", path=str(path), edges=edge_index.shape[1])
        
        return edge_index
