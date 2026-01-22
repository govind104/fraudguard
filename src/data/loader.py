"""Memory-optimized data loading with stratified sampling.

Provides the FraudDataLoader class for loading IEEE-CIS fraud detection data
with configurable sampling and memory-efficient dtypes.

Example:
    >>> from src.data.loader import FraudDataLoader
    >>> loader = FraudDataLoader()
    >>> df = loader.load_train_data(sample_frac=0.1)
    >>> print(f"Loaded {len(df)} rows, fraud ratio: {df.isFraud.mean():.2%}")
"""

from typing import Optional, Tuple

import pandas as pd

from src.utils.config import DataConfig, load_data_config
from src.utils.exceptions import DataLoadingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudDataLoader:
    """IEEE-CIS fraud detection data loader with stratified sampling.
    
    Loads transaction data with memory-optimized dtypes and supports
    stratified sampling to maintain fraud ratio in smaller datasets.
    
    Attributes:
        config: DataConfig with paths and settings.
        
    Example:
        >>> loader = FraudDataLoader()
        >>> train_df = loader.load_train_data(sample_frac=0.1)
        >>> test_df = loader.load_test_data()
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize data loader.
        
        Args:
            config: Optional DataConfig. Loads from default if not provided.
        """
        self.config = config or load_data_config()
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required data files exist.
        
        Raises:
            DataLoadingError: If raw data directory doesn't exist.
        """
        if not self.config.paths.raw_data_dir.exists():
            raise DataLoadingError(
                "Raw data directory not found",
                path=str(self.config.paths.raw_data_dir),
                hint="Ensure ieee-fraud-detection folder exists at ../ieee-fraud-detection/",
            )
    
    def _get_dtypes(self) -> dict:
        """Convert config dtypes to pandas-compatible format.
        
        Returns:
            Dictionary mapping column names to pandas dtypes.
        """
        dtype_map = {}
        for col, dtype in self.config.dtypes.items():
            if dtype == "str":
                dtype_map[col] = str
            elif dtype == "category":
                dtype_map[col] = "category"
            else:
                dtype_map[col] = dtype
        return dtype_map
    
    def load_train_data(
        self,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Load training transaction data with optional stratified sampling.
        
        Uses stratified sampling to maintain the original fraud ratio
        when loading a subset of the data for development.
        
        Args:
            sample_frac: Fraction of data to load (0.0-1.0). 
                        None uses config default. 1.0 loads full dataset.
            random_state: Random seed for reproducibility.
            
        Returns:
            DataFrame with transaction features and 'isFraud' label.
            
        Raises:
            DataLoadingError: If train_transaction.csv not found.
            
        Example:
            >>> loader = FraudDataLoader()
            >>> df = loader.load_train_data(sample_frac=0.1)
            >>> print(f"Fraud ratio: {df.isFraud.mean():.2%}")
            Fraud ratio: 3.50%
        """
        file_path = (
            self.config.paths.raw_data_dir / 
            self.config.files["train_transaction"]
        )
        
        if not file_path.exists():
            raise DataLoadingError(
                "Training data file not found",
                path=str(file_path),
            )
        
        sample_frac = sample_frac or self.config.sampling["default_frac"]
        
        logger.info(
            "Loading training data",
            path=str(file_path),
            sample_frac=sample_frac,
        )
        
        # Load with optimized dtypes
        df = pd.read_csv(
            file_path,
            dtype=self._get_dtypes(),
            usecols=self.config.features["columns"],
        )
        
        logger.info(
            "Raw data loaded",
            rows=len(df),
            fraud_count=int(df["isFraud"].sum()),
            fraud_ratio=f"{df['isFraud'].mean():.2%}",
        )
        
        # Apply stratified sampling if fraction < 1.0
        if sample_frac < 1.0:
            df = self._stratified_sample(df, sample_frac, random_state)
        
        return df
    
    def _stratified_sample(
        self,
        df: pd.DataFrame,
        sample_frac: float,
        random_state: int,
    ) -> pd.DataFrame:
        """Apply stratified sampling maintaining fraud ratio.
        
        Args:
            df: Full DataFrame.
            sample_frac: Fraction to sample.
            random_state: Random seed.
            
        Returns:
            Sampled DataFrame with preserved fraud ratio.
        """
        fraud = df[df["isFraud"] == 1].sample(frac=sample_frac, random_state=random_state)
        non_fraud = df[df["isFraud"] == 0].sample(frac=sample_frac, random_state=random_state)
        
        sampled = pd.concat([fraud, non_fraud]).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
        
        logger.info(
            "Stratified sampling complete",
            original_rows=len(df),
            sampled_rows=len(sampled),
            fraud_ratio=f"{sampled['isFraud'].mean():.2%}",
        )
        
        return sampled
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test transaction data.
        
        Returns:
            DataFrame with transaction features (no labels).
            
        Raises:
            DataLoadingError: If test_transaction.csv not found.
        """
        file_path = (
            self.config.paths.raw_data_dir / 
            self.config.files["test_transaction"]
        )
        
        if not file_path.exists():
            raise DataLoadingError(
                "Test data file not found",
                path=str(file_path),
            )
        
        # Remove isFraud from columns for test data
        columns = [c for c in self.config.features["columns"] if c != "isFraud"]
        dtypes = {k: v for k, v in self._get_dtypes().items() if k != "isFraud"}
        
        logger.info("Loading test data", path=str(file_path))
        
        df = pd.read_csv(file_path, dtype=dtypes, usecols=columns)
        
        logger.info("Test data loaded", rows=len(df))
        
        return df
    
    def create_splits(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits using chronological ordering.
        
        Sorts by TransactionDT and splits according to config ratios.
        This maintains temporal ordering for realistic evaluation.
        
        Args:
            df: Full DataFrame with isFraud labels.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
            
        Example:
            >>> train, val, test = loader.create_splits(df)
            >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        """
        # Sort by transaction time
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        
        train_ratio = self.config.splits["train_ratio"]
        val_ratio = self.config.splits["val_ratio"]
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(
            "Created chronological splits",
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            train_fraud_ratio=f"{train_df['isFraud'].mean():.2%}",
        )
        
        return train_df, val_df, test_df
