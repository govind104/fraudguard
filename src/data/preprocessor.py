"""Feature engineering pipeline with artifact persistence.

Provides the FeaturePreprocessor class for transforming raw transaction data
into normalized feature tensors with PCA dimensionality reduction.

Example:
    >>> from src.data.preprocessor import FeaturePreprocessor
    >>> preprocessor = FeaturePreprocessor()
    >>> X = preprocessor.fit_transform(train_df)
    >>> preprocessor.save_artifacts()
    >>> # Later: X_test = preprocessor.transform(test_df)
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.config import DataConfig, ModelConfig, load_data_config, load_model_config
from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePreprocessor:
    """Feature engineering pipeline with artifact persistence.
    
    Transforms raw transaction data through:
    1. Missing value imputation
    2. Temporal feature engineering (hour, time since last)
    3. Categorical feature hashing
    4. StandardScaler normalization
    5. PCA dimensionality reduction
    
    Fitted scaler and PCA can be saved/loaded for inference consistency.
    
    Attributes:
        scaler: Fitted StandardScaler (None until fit).
        pca: Fitted PCA transformer (None until fit).
        is_fitted: Whether fit_transform has been called.
        
    Example:
        >>> prep = FeaturePreprocessor()
        >>> X_train = prep.fit_transform(train_df)
        >>> X_val = prep.transform(val_df)
        >>> prep.save_artifacts("data/processed/")
    """
    
    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        """Initialize preprocessor.
        
        Args:
            data_config: Optional DataConfig. Loads from default if not provided.
            model_config: Optional ModelConfig for preprocessing params.
        """
        self.data_config = data_config or load_data_config()
        self.model_config = model_config or load_model_config()
        
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.is_fitted: bool = False
        
        # Store feature dimensions for validation
        self._num_features: Optional[int] = None
        self._pca_components: Optional[int] = None
    
    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Fit preprocessing pipeline and transform data.
        
        Must be called on training data first. Fits scaler and PCA,
        then transforms features to tensor.
        
        Args:
            df: Training DataFrame with transaction features.
            
        Returns:
            Float32 tensor of shape (n_samples, n_components).
            
        Raises:
            PreprocessingError: If feature engineering fails.
            
        Example:
            >>> prep = FeaturePreprocessor()
            >>> X_train = prep.fit_transform(train_df)
            >>> print(f"Shape: {X_train.shape}")
        """
        logger.info("Starting fit_transform", rows=len(df))
        
        try:
            # Step 1: Feature engineering
            features = self._engineer_features(df)
            
            # Step 2: Fit and apply StandardScaler
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            
            # Step 3: Fit and apply PCA
            pca_variance = self.model_config.preprocessing["pca_variance"]
            self.pca = PCA(n_components=pca_variance)
            features_pca = self.pca.fit_transform(features_scaled)
            
            self._num_features = features.shape[1]
            self._pca_components = features_pca.shape[1]
            self.is_fitted = True
            
            logger.info(
                "Fit complete",
                input_features=self._num_features,
                pca_components=self._pca_components,
                variance_explained=f"{sum(self.pca.explained_variance_ratio_):.2%}",
            )
            
            return torch.tensor(features_pca, dtype=torch.float32)
            
        except Exception as e:
            raise PreprocessingError(
                f"Feature engineering failed: {str(e)}",
                stage="fit_transform",
            )
    
    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform data using fitted pipeline.
        
        Must be called after fit_transform. Uses previously fitted
        scaler and PCA for consistent transformation.
        
        Args:
            df: DataFrame with transaction features.
            
        Returns:
            Float32 tensor of shape (n_samples, n_components).
            
        Raises:
            PreprocessingError: If not fitted or transformation fails.
        """
        if not self.is_fitted:
            raise PreprocessingError(
                "Preprocessor not fitted. Call fit_transform first.",
                stage="transform",
            )
        
        logger.info("Transforming data", rows=len(df))
        
        try:
            features = self._engineer_features(df)
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)
            
            return torch.tensor(features_pca, dtype=torch.float32)
            
        except Exception as e:
            raise PreprocessingError(
                f"Transformation failed: {str(e)}",
                stage="transform",
            )
    
    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features from raw data.
        
        Creates:
        - Temporal features (TransactionHour, TimeSinceLast)
        - Hashed categorical features (email, card4)
        - Numeric features
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            NumPy array of engineered features.
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown")
        df["card4"] = df["card4"].fillna("unknown")
        
        # Temporal features
        df["TransactionHour"] = (df["TransactionDT"] // 3600) % 24
        df = df.sort_values("TransactionDT")
        df["TimeSinceLast"] = df["TransactionDT"].diff().fillna(0)
        
        # Numeric features
        num_cols = ["TransactionAmt", "C1", "C2", "C3", "TransactionHour", "TimeSinceLast"]
        num_features = df[num_cols].fillna(0).values
        
        # Hash categorical features
        email_buckets = self.model_config.preprocessing["email_hash_buckets"]
        card_buckets = self.model_config.preprocessing["card_hash_buckets"]
        
        email_hash = pd.get_dummies(
            pd.util.hash_pandas_object(df["P_emaildomain"]) % email_buckets,
            prefix="email",
        ).values
        
        card_hash = pd.get_dummies(
            pd.util.hash_pandas_object(df["card4"]) % card_buckets,
            prefix="card",
        ).values
        
        # Combine all features
        features = np.hstack([num_features, email_hash, card_hash])
        
        return features.astype(np.float32)
    
    def save_artifacts(self, path: Optional[Path] = None) -> None:
        """Save fitted scaler and PCA to disk.
        
        Args:
            path: Directory to save artifacts. Defaults to config processed_dir.
            
        Raises:
            PreprocessingError: If not fitted or save fails.
        """
        if not self.is_fitted:
            raise PreprocessingError(
                "Cannot save: preprocessor not fitted",
                stage="save_artifacts",
            )
        
        if path is None:
            path = self.data_config.paths.processed_dir
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        scaler_path = path / "scaler.pkl"
        pca_path = path / "pca.pkl"
        
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(pca_path, "wb") as f:
            pickle.dump(self.pca, f)
        
        logger.info(
            "Artifacts saved",
            scaler_path=str(scaler_path),
            pca_path=str(pca_path),
        )
    
    def load_artifacts(self, path: Optional[Path] = None) -> None:
        """Load fitted scaler and PCA from disk.
        
        Args:
            path: Directory containing artifacts. Defaults to config processed_dir.
            
        Raises:
            PreprocessingError: If artifacts not found.
        """
        if path is None:
            path = self.data_config.paths.processed_dir
        
        path = Path(path)
        scaler_path = path / "scaler.pkl"
        pca_path = path / "pca.pkl"
        
        if not scaler_path.exists() or not pca_path.exists():
            raise PreprocessingError(
                "Artifacts not found",
                path=str(path),
            )
        
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        
        self.is_fitted = True
        self._pca_components = self.pca.n_components_
        
        logger.info("Artifacts loaded", path=str(path))
