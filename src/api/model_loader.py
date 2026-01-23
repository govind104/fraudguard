"""Model loader for FraudGuard API.

Handles loading trained GNN model from MLflow or disk,
with caching and thread-safety for production use.

Example:
    >>> loader = ModelLoader()
    >>> model, preprocessor = loader.load()
    >>> predictions = model(features, edge_index)
"""

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import faiss
import numpy as np
import torch

from src.data.graph_builder import GraphBuilder
from src.data.preprocessor import FeaturePreprocessor
from src.models.gnn import FraudGNN
from src.utils.config import ModelConfig, load_model_config
from src.utils.device_utils import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thread lock for model loading
_model_lock = threading.Lock()


class ModelLoader:
    """Thread-safe model loader with caching.

    Loads trained FraudGNN model from:
    1. MLflow Model Registry (if available)
    2. Local checkpoint file (fallback)

    Attributes:
        model: Loaded FraudGNN model
        preprocessor: Feature preprocessor
        graph_builder: Graph construction (for inference)
        device: Compute device

    Example:
        >>> loader = ModelLoader(model_path="models/fraudguard_AD_RL.pt")
        >>> model = loader.get_model()
        >>> pred = model(x, edge_index)
    """

    _instance: Optional["ModelLoader"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for model loader."""
        if cls._instance is None:
            with _model_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize model loader.

        Args:
            model_path: Path to model checkpoint (.pt file)
            config: Model configuration
            device: Compute device (auto-detected if None)
        """
        # Skip re-initialization if already loaded
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_path = model_path or os.environ.get(
            "FRAUDGUARD_MODEL_PATH", "models/processed/fraudguard_AD_RL.pt"
        )
        self.config = config or load_model_config()
        self.device = device or get_device()

        # Components (loaded lazily)
        self._model: Optional[FraudGNN] = None
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._graph_builder: Optional[GraphBuilder] = None

        # RAG artifacts (loaded lazily)
        self._faiss_index: Optional[faiss.Index] = None
        self._feature_store: Optional[np.ndarray] = None
        self._index_to_id: Optional[np.ndarray] = None

        # Metadata
        self.model_version: str = "1.0.0"
        self.gmeans: float = 0.5724  # From A/B test
        self.training_date: str = "2026-01-22"

        self._initialized = True
        logger.info(f"ModelLoader initialized with path: {self.model_path}")

    def _load_model(self) -> FraudGNN:
        """Load FraudGNN from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        # Try MLflow first
        try:
            import mlflow.pytorch

            # Try to load from MLflow registry
            model_uri = "models:/FraudGuard-Production/latest"
            model = mlflow.pytorch.load_model(model_uri)
            self.model_version = "mlflow-latest"
            logger.info("Loaded model from MLflow registry")
            return model.to(self.device)

        except Exception as e:
            logger.warning(f"MLflow load failed: {e}, falling back to checkpoint")

        # Fallback to local checkpoint
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first or set FRAUDGUARD_MODEL_PATH."
            )

        # Determine input dimension from preprocessor
        input_dim = self._get_input_dim()

        # Create model architecture
        model = FraudGNN(
            in_channels=input_dim,
            hidden_channels=self.config.gnn["hidden_channels"],
            dropout=self.config.gnn["dropout"],
            use_batch_norm=self.config.gnn.get("batch_norm", True),
        ).to(self.device)

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        logger.info(f"Model loaded from checkpoint: {self.model_path}")
        return model

    def _get_input_dim(self) -> int:
        """Get input dimension from preprocessor or default."""
        try:
            preprocessor = self.get_preprocessor()
            if hasattr(preprocessor, "pca") and preprocessor.pca is not None:
                return preprocessor.pca.n_components_
        except Exception:
            pass

        # Default based on typical IEEE-CIS preprocessing
        return 32  # After PCA

    def _load_preprocessor(self) -> FeaturePreprocessor:
        """Load fitted preprocessor."""
        preprocessor = FeaturePreprocessor(config=self.config)

        # Try multiple paths for preprocessor artifacts
        # Priority: env var > models/processed > data/processed
        processed_dir = os.environ.get("FRAUDGUARD_PROCESSED_DIR", None)

        search_paths = []
        if processed_dir:
            search_paths.append(Path(processed_dir))

        # Derive from model path (e.g., models/fraudguard_AD_RL.pt -> models/processed/)
        model_dir = Path(self.model_path).parent
        search_paths.append(model_dir / "processed")

        # Fallback paths
        search_paths.extend(
            [
                Path("models/processed"),
                Path("data/processed"),
                Path("/app/models/processed"),  # Docker path
                Path("/app/data/processed"),  # Docker fallback
            ]
        )

        # Find first valid path
        import pickle

        for base_path in search_paths:
            scaler_path = base_path / "scaler.pkl"
            pca_path = base_path / "pca.pkl"

            if scaler_path.exists() and pca_path.exists():
                with open(scaler_path, "rb") as f:
                    preprocessor.scaler = pickle.load(f)
                with open(pca_path, "rb") as f:
                    preprocessor.pca = pickle.load(f)
                logger.info(f"Loaded preprocessor from {base_path}")
                return preprocessor

        logger.warning(
            "Preprocessor files not found in any search path, using unfitted preprocessor"
        )
        return preprocessor

    def _load_graph_builder(self) -> GraphBuilder:
        """Load graph builder for inference."""
        return GraphBuilder(config=self.config)

    def _load_rag_artifacts(self) -> None:
        """Load RAG artifacts (FAISS index, feature store, ID mapping).

        These artifacts enable retrieval-augmented inference by providing
        k-NN neighbor lookup for incoming transactions.
        """
        # Try multiple paths for RAG artifacts
        processed_dir = os.environ.get("FRAUDGUARD_PROCESSED_DIR", None)

        search_paths = []
        if processed_dir:
            search_paths.append(Path(processed_dir))

        # Standard paths
        search_paths.extend(
            [
                Path("models/processed"),
                Path("/app/models/processed"),  # Docker path
            ]
        )

        for base_path in search_paths:
            faiss_path = base_path / "faiss.index"
            store_path = base_path / "feature_store.npy"
            map_path = base_path / "index_to_id.npy"

            if faiss_path.exists() and store_path.exists():
                # Load FAISS index
                self._faiss_index = faiss.read_index(str(faiss_path))
                logger.info(
                    f"[OK] Loaded FAISS index with {self._faiss_index.ntotal:,} vectors"
                )

                # Load feature store (memory-mapped for efficiency)
                self._feature_store = np.load(str(store_path), mmap_mode="r")
                logger.info(
                    f"[OK] Loaded feature store: shape {self._feature_store.shape}"
                )

                # Load ID mapping (optional, for explainability)
                if map_path.exists():
                    self._index_to_id = np.load(str(map_path))
                    logger.info("[OK] Loaded ID mapping for explainability")

                return

        logger.warning(
            "[WARNING] RAG artifacts not found. Run build_inference_artifacts.py first."
        )

    def get_faiss_index(self) -> Optional[faiss.Index]:
        """Get FAISS index for k-NN search (thread-safe, cached)."""
        if self._faiss_index is None:
            with _model_lock:
                if self._faiss_index is None:
                    self._load_rag_artifacts()
        return self._faiss_index

    def get_feature_store(self) -> Optional[np.ndarray]:
        """Get feature store for neighbor feature retrieval."""
        if self._feature_store is None:
            with _model_lock:
                if self._feature_store is None:
                    self._load_rag_artifacts()
        return self._feature_store

    def get_index_to_id(self) -> Optional[np.ndarray]:
        """Get ID mapping for explainability."""
        if self._index_to_id is None:
            with _model_lock:
                if self._index_to_id is None:
                    self._load_rag_artifacts()
        return self._index_to_id

    def get_model(self) -> FraudGNN:
        """Get loaded model (thread-safe, cached)."""
        if self._model is None:
            with _model_lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def get_preprocessor(self) -> FeaturePreprocessor:
        """Get loaded preprocessor (thread-safe, cached)."""
        if self._preprocessor is None:
            with _model_lock:
                if self._preprocessor is None:
                    self._preprocessor = self._load_preprocessor()
        return self._preprocessor

    def get_graph_builder(self) -> GraphBuilder:
        """Get graph builder (thread-safe, cached)."""
        if self._graph_builder is None:
            with _model_lock:
                if self._graph_builder is None:
                    self._graph_builder = self._load_graph_builder()
        return self._graph_builder

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_name": "FraudGuard-GNN",
            "version": self.model_version,
            "gmeans": self.gmeans,
            "training_date": self.training_date,
            "device": str(self.device),
            "model_path": self.model_path,
        }

    def reload(self) -> None:
        """Force reload model (useful for hot reloading)."""
        with _model_lock:
            self._model = None
            self._preprocessor = None
            self._model = self._load_model()
            self._preprocessor = self._load_preprocessor()
            logger.info("Model reloaded")


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    """Get singleton model loader instance."""
    return ModelLoader()
