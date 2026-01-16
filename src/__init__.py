"""FraudGuard: Production-ready GNN fraud detection system.

This package provides a modular implementation of the AD-RL-GNN fraud detection
framework with:
- Adaptive Majority Class Downsampling (AdaptiveMCD)
- Reinforcement Learning-driven subgraph selection
- FAISS-accelerated semantic similarity graph construction
- Graph Neural Network classification with Focal Loss

Example:
    >>> from src.data.loader import FraudDataLoader
    >>> from src.data.preprocessor import FeaturePreprocessor
    >>> loader = FraudDataLoader()
    >>> df = loader.load_train_data(sample_frac=0.1)
    >>> preprocessor = FeaturePreprocessor()
    >>> X = preprocessor.fit_transform(df)
"""

__version__ = "0.1.0"
