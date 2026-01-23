"""Build inference artifacts for RAG-augmented GNN inference.

Generates the "Knowledge Base" consisting of:
- FAISS Index for k-NN neighbor search
- Feature Store (memory-mapped numpy array)
- ID Mapping for explainability (FAISS index -> TransactionID)

CRITICAL: Only indexes Training set (first 80% by time) to prevent data leakage.

Usage:
    cd fraudguard
    uv run python scripts/build_inference_artifacts.py
"""

import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import FeaturePreprocessor
from src.utils.config import load_data_config, load_model_config
from src.utils.device_utils import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_artifacts() -> None:
    """Build FAISS index and feature store from training data."""
    set_seed(42)
    logger.info("[STARTUP] Starting inference artifact generation...")

    # 1. Configuration
    config = load_data_config()
    model_config = load_model_config()

    # Force parity with training threshold
    similarity_threshold = model_config.graph.similarity_threshold
    logger.info(f"Using similarity_threshold={similarity_threshold}")

    raw_path = config.paths.raw_data_dir / "train_transaction.csv"
    processed_dir = Path("models/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data (Strict Temporal Split)
    logger.info(f"Loading data from {raw_path}")

    # Load only necessary columns for memory efficiency
    needed_cols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "C1",
        "C2",
        "C3",
        "P_emaildomain",
        "card4",
    ]

    if not raw_path.exists():
        logger.error(f"[ERROR] Dataset not found at {raw_path}")
        logger.error("Please ensure IEEE-CIS dataset is placed at ../ieee-fraud-detection/")
        sys.exit(1)

    df = pd.read_csv(raw_path, usecols=needed_cols)
    logger.info(f"Loaded {len(df):,} transactions")

    # SORT by Time (Crucial for temporal validity)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # SPLIT: Keep only first 80% (Training Set) for the Index
    # This prevents data leakage from "future" transactions
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()

    logger.info(
        f"Temporal split: indexing first {len(df_train):,} rows (80%) to prevent leakage"
    )

    # 3. Load FROZEN preprocessor artifacts (Must exist from Colab training)
    preprocessor = FeaturePreprocessor()

    scaler_path = processed_dir / "scaler.pkl"
    pca_path = processed_dir / "pca.pkl"

    if not scaler_path.exists() or not pca_path.exists():
        logger.error("[ERROR] Missing preprocessor artifacts!")
        logger.error(f"  Expected: {scaler_path}")
        logger.error(f"  Expected: {pca_path}")
        logger.error("You MUST run Colab training to generate scaler.pkl and pca.pkl first.")
        sys.exit(1)

    preprocessor.load_artifacts(processed_dir)
    logger.info("[OK] Loaded frozen preprocessor artifacts (Scaler + PCA)")

    # 4. Transform data
    logger.info("Transforming training data...")
    features = preprocessor.transform(df_train)

    # Ensure we have numpy array
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = np.array(features, dtype=np.float32)

    logger.info(f"Feature shape: {features_np.shape} (76 raw -> 69 after PCA)")

    # 5. Save ID Mapping (for explainability)
    # Maps internal FAISS ID (0..N) -> Original TransactionID
    id_map = df_train["TransactionID"].values.astype(np.int64)
    id_map_path = processed_dir / "index_to_id.npy"
    np.save(id_map_path, id_map)
    logger.info(f"[OK] Saved ID map for {len(id_map):,} nodes -> {id_map_path}")

    # 6. Build FAISS Index
    logger.info("Building FAISS index...")
    d = features_np.shape[1]  # Should be 32 after PCA
    index = faiss.IndexFlatL2(d)
    index.add(features_np.astype(np.float32))
    logger.info(f"[OK] FAISS index built with {index.ntotal:,} vectors of dimension {d}")

    # 7. Save Artifacts
    faiss_path = processed_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"[OK] Saved FAISS index -> {faiss_path}")

    # Save Feature Store (Float32 for performance)
    feature_store_path = processed_dir / "feature_store.npy"
    np.save(feature_store_path, features_np.astype(np.float32))
    logger.info(f"[OK] Saved feature store -> {feature_store_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("[OK] Artifact generation complete!")
    logger.info(f"  - FAISS Index:    {faiss_path} ({faiss_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(
        f"  - Feature Store:  {feature_store_path} ({feature_store_path.stat().st_size / 1e6:.1f} MB)"
    )
    logger.info(f"  - ID Map:         {id_map_path}")
    logger.info(f"  - Indexed Nodes:  {index.ntotal:,}")
    logger.info(f"  - Feature Dim:    {d}")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_artifacts()
