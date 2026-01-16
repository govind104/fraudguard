# FraudGuard: Production-Ready GNN Fraud Detection

A production-grade fraud detection system built with Graph Neural Networks, Reinforcement Learning-driven subgraph selection, and Adaptive Majority Class Downsampling.

## Features

- **GNN-based Classification**: 3-layer GCN with batch normalization and dropout
- **RL Subgraph Selection**: Dynamic choice of Random Walk, K-hop, or K-ego neighborhoods  
- **Adaptive MCD**: Learned downsampling for class imbalance
- **Leak-Free Graph Construction**: Training data only for edge building
- **FAISS Similarity Edges**: Scalable semantic similarity graph construction

## Installation

```bash
cd fraudguard
poetry install
```

## Quick Start

```python
from src.data.loader import FraudDataLoader
from src.data.preprocessor import FeaturePreprocessor
from src.data.graph_builder import GraphBuilder

# Load data
loader = FraudDataLoader()
df = loader.load_train_data(sample_frac=0.1)

# Preprocess
preprocessor = FeaturePreprocessor()
X = preprocessor.fit_transform(df)

# Build graph (training only)
builder = GraphBuilder()
edge_index = builder.fit(X)
```

## Project Structure

```
fraudguard/
├── config/           # YAML configuration files
├── data/
│   ├── processed/    # Saved artifacts (scaler, PCA)
│   └── graphs/       # Cached edge indices
├── src/
│   ├── data/         # Data loading, preprocessing, graph building
│   ├── models/       # AdaptiveMCD, RL Agent, MCES, GNN
│   ├── training/     # Training orchestration
│   ├── inference/    # Prediction engine
│   └── utils/        # Logging, device utils, exceptions
└── tests/            # Unit and integration tests
```

## Data

Place IEEE-CIS fraud detection data in `../ieee-fraud-detection/`:
- `train_transaction.csv` (590,540 rows)
- `test_transaction.csv` (for evaluation)

## Configuration

Edit `config/data_config.yaml` and `config/model_config.yaml` to customize:
- Data paths and sampling fractions
- Model hyperparameters
- Training settings

## License

MIT
