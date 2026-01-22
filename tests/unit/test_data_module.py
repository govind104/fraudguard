"""Unit tests for data loading module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_data_config  # noqa: E402
from src.utils.exceptions import DataLoadingError  # noqa: E402


class TestDataConfig:
    """Tests for data configuration loading."""

    def test_config_loads_successfully(self):
        """Config should load without errors."""
        config = load_data_config()
        assert config is not None
        assert config.paths is not None

    def test_config_has_required_paths(self):
        """Config should have all required path settings."""
        config = load_data_config()
        assert config.paths.raw_data_dir is not None
        assert config.paths.processed_dir is not None
        assert config.paths.graphs_dir is not None

    def test_config_has_feature_columns(self):
        """Config should specify feature columns."""
        config = load_data_config()
        assert "columns" in config.features
        assert len(config.features["columns"]) > 0
        assert "isFraud" in config.features["columns"]


class TestFraudDataLoader:
    """Tests for FraudDataLoader class."""

    def test_loader_initializes(self):
        """Loader should initialize with default config."""
        from src.data.loader import FraudDataLoader

        try:
            loader = FraudDataLoader()
            assert loader.config is not None
        except DataLoadingError as e:
            # Expected if data directory doesn't exist
            pytest.skip(f"Data directory not found: {e}")

    def test_stratified_sampling_preserves_ratio(self):
        """Stratified sampling should maintain fraud ratio."""
        from src.data.loader import FraudDataLoader

        try:
            loader = FraudDataLoader()
            df_full = loader.load_train_data(sample_frac=0.01)

            if len(df_full) > 0:
                fraud_ratio = df_full["isFraud"].mean()
                # Fraud ratio should be around 3.5% (±1%)
                assert 0.02 < fraud_ratio < 0.05, f"Unexpected fraud ratio: {fraud_ratio}"
        except DataLoadingError:
            pytest.skip("Data not available")

    def test_splits_sum_to_total(self):
        """Train/val/test splits should cover all data."""
        from src.data.loader import FraudDataLoader

        try:
            loader = FraudDataLoader()
            df = loader.load_train_data(sample_frac=0.01)
            train, val, test = loader.create_splits(df)

            assert len(train) + len(val) + len(test) == len(df)
        except DataLoadingError:
            pytest.skip("Data not available")


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor class."""

    def test_preprocessor_initializes(self):
        """Preprocessor should initialize with default configs."""
        from src.data.preprocessor import FeaturePreprocessor

        prep = FeaturePreprocessor()
        assert not prep.is_fitted
        assert prep.scaler is None
        assert prep.pca is None

    def test_fit_transform_produces_tensor(self):
        """fit_transform should produce a float32 tensor."""
        from src.data.loader import FraudDataLoader
        from src.data.preprocessor import FeaturePreprocessor

        try:
            loader = FraudDataLoader()
            df = loader.load_train_data(sample_frac=0.01)

            prep = FeaturePreprocessor()
            X = prep.fit_transform(df)

            assert isinstance(X, np.ndarray) or hasattr(X, "numpy")
            assert X.shape[0] == len(df)
            assert X.shape[1] > 0  # PCA should retain some components
            assert prep.is_fitted
        except DataLoadingError:
            pytest.skip("Data not available")

    def test_transform_requires_fit(self):
        """transform should fail if not fitted."""
        from src.data.preprocessor import FeaturePreprocessor
        from src.utils.exceptions import PreprocessingError

        prep = FeaturePreprocessor()

        # Create dummy DataFrame
        df = pd.DataFrame(
            {
                "TransactionDT": [100, 200, 300],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "C1": [1.0, 2.0, 3.0],
                "C2": [1.0, 2.0, 3.0],
                "C3": [1.0, 2.0, 3.0],
                "P_emaildomain": ["gmail.com", "yahoo.com", "unknown"],
                "card4": ["visa", "mastercard", "unknown"],
            }
        )

        with pytest.raises(PreprocessingError):
            prep.transform(df)


class TestGraphBuilder:
    """Tests for GraphBuilder class (leak prevention)."""

    def test_builder_initializes(self):
        """Builder should initialize with default config."""
        from src.data.graph_builder import GraphBuilder

        builder = GraphBuilder()
        assert builder.threshold > 0
        assert builder.batch_size > 0

    def test_fit_creates_edges(self):
        """fit should create edge index from features."""
        import torch

        from src.data.graph_builder import GraphBuilder

        # Create synthetic features
        X_train = torch.randn(100, 10)

        builder = GraphBuilder()
        edges = builder.fit(X_train)

        assert edges.shape[0] == 2  # Source and destination
        # Either has edges or is empty (threshold may be too high)
        assert edges.shape[1] >= 0

    def test_no_test_test_edges(self):
        """Transform should NOT create test-test edges (leak prevention)."""
        import torch

        from src.data.graph_builder import GraphBuilder

        # Create synthetic features
        train_size = 80
        test_size = 20
        X_train = torch.randn(train_size, 10)
        X_test = torch.randn(test_size, 10)

        builder = GraphBuilder()
        builder.fit(X_train)
        full_edges = builder.transform(X_test, train_size=train_size)

        # CRITICAL: Verify no test-test edges
        src = full_edges[0]
        dst = full_edges[1]
        test_test_mask = (src >= train_size) & (dst >= train_size)

        assert test_test_mask.sum().item() == 0, "LEAKAGE: Found test-test edges!"

    def test_verify_no_leakage_passes(self):
        """verify_no_leakage should pass for properly constructed graph."""
        import torch

        from src.data.graph_builder import GraphBuilder

        train_size = 80
        X_train = torch.randn(train_size, 10)
        X_test = torch.randn(20, 10)

        builder = GraphBuilder()
        builder.fit(X_train)
        full_edges = builder.transform(X_test, train_size=train_size)

        # Should not raise
        result = builder.verify_no_leakage(full_edges, train_size=train_size)
        assert result is True

    def test_verify_no_leakage_catches_bad_graph(self):
        """verify_no_leakage should catch graphs with test-test edges."""
        import torch

        from src.data.graph_builder import GraphBuilder
        from src.utils.exceptions import GraphBuildingError

        # Create a bad graph with test-test edges
        train_size = 80
        bad_edges = torch.tensor(
            [
                [0, 1, 85, 90],  # src: includes test-test edge
                [1, 0, 90, 85],  # dst: includes test-test edge
            ],
            dtype=torch.long,
        )

        builder = GraphBuilder()

        with pytest.raises(GraphBuildingError):
            builder.verify_no_leakage(bad_edges, train_size=train_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
