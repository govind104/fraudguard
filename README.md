# FraudGuard: Production-Ready GNN Fraud Detection

A production-grade fraud detection system built with Graph Neural Networks, Reinforcement Learning-driven subgraph selection, and Adaptive Majority Class Downsampling.

## Features

- **GNN-based Classification**: 3-layer GCN with batch normalization and dropout
- **RL Subgraph Selection**: Dynamic choice of Random Walk, K-hop, or K-ego neighborhoods  
- **Adaptive MCD**: Learned downsampling for 28:1 class imbalance
- **Leak-Free Graph Construction**: Strict temporal splitting (80/20) for edge building
- **FAISS Similarity Edges**: Scalable semantic similarity graph construction
- **Retrieval-Augmented Inference (RAG)**: Real-time dynamic graph construction using FAISS k-NN (~50ms latency)
- **MLflow Integration**: Experiment tracking, parameter logging, and model versioning

## Installation

```bash
cd fraudguard
# Install dependencies with uv (fast)
uv sync
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
│   │   ├── scaler.pkl          ← Frozen Scaler
│   │   ├── pca.pkl             ← Frozen PCA
│   │   ├── faiss.index         ← RAG Knowledge Base (472k vectors)
│   │   ├── feature_store.npy   ← Mmap-able Feature Store
│   │   └── index_to_id.npy     ← Explainability Mapping
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

## Reproducing Results

### Quick Test (10% data, ~5 min)

```bash
python scripts/train.py --sample_frac 0.1 --epochs 10
```

### Full Training (100% data, ~2 hours)

```bash
python scripts/train.py --sample_frac 1.0 --epochs 30
```

### Skip Baseline (faster, AD-RL-GNN only)

```bash
python scripts/train.py --sample_frac 1.0 --skip_baseline
```

### Verified Results (A/B Test)

| Metric | Baseline GNN | AD-RL-GNN | Improvement |
|--------|--------------|-----------|-------------|
| G-Means | 46.61% | 57.21% | **+22.7%** |
| P95 Latency | 26.86ms | 27.84ms | ~Same |

## Design Choices

### Intentional Minimal Feature Engineering

This project deliberately uses minimal feature engineering (no user-aggregations, no identity table joins) to **isolate the architectural contribution** of the AD-RL-GNN framework.

The ~23% G-Means improvement over baseline is purely from:

- Adaptive Majority Class Downsampling (MCD)
- RL-based subgraph selection for fraud pattern recovery
- 3-layer GCN with batch normalization

For comparison, top Kaggle solutions ([Artgor's approach](https://www.kaggle.com/artgor/eda-and-models)) achieve ~96% AUC through extensive feature engineering including:

- User ID proxies (card1 + addr1 + D1)
- Transaction velocity aggregations
- Device/identity table joins

**Production deployments** would incorporate domain-specific feature engineering on top of this architecture to maximize absolute performance.

## Google Colab Training

For full dataset training (91M+ edges), use Google Colab with GPU:

1. **Push to GitHub** (ensure `data/` is in `.gitignore`)
2. **Open Colab** and follow `notebooks/colab_training.md`
3. **Key steps:**

   ```python
   # Mount Drive for data
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Clone and install
   !git clone https://github.com/YOUR_USERNAME/fraudguard.git
   !pip install -e fraudguard
   !pip install faiss-gpu  # GPU FAISS for fast indexing
   ```

The notebook uses `NeighborLoader` for mini-batch training, enabling training on graphs too large for local memory.

## Production API

### Quick Start

```bash
# Generates the FAISS index and feature store from training data (inference artifacts required for RAG)
uv run python scripts/build_inference_artifacts.py

# Start API with Docker Compose
docker-compose up -d

# Test health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TX123", "features": {"TransactionAmt": 150.0, "card4": "visa"}}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for load balancers |
| `/model/info` | GET | Model metadata and metrics |
| `/predict` | POST | Single transaction prediction |
| `/predict/batch` | POST | Batch predictions (max 100) |
| `/model/reload` | POST | Hot reload model (admin) |

### GNNExplainer Integration

Every fraud prediction includes regulatory-compliant feature attributions:

> **Note:** Requires `torch-scatter` and `torch-sparse` (installed via PyG wheels). On local CPU environments without these libraries, explanations will gracefully degrade to `null`.

```json
{
  "is_fraud": true,
  "fraud_probability": 0.87,
  "explanation": {
    "top_features": [
      {"name": "TransactionAmt", "importance": 0.42, "direction": "positive"},
      {"name": "TimeSinceLast", "importance": 0.28, "direction": "negative"}
    ],
    "subgraph_nodes": [42, 156, 892]
  }
}
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI fraud detection |
| Redis | 6379 | Prediction caching |
| MLflow | 5000 | Experiment tracking |

## Local Training

### Command Line Options

```bash
python scripts/train.py --help

# Key options:
#   --data_dir       Path to IEEE-CIS data (default: ../ieee-fraud-detection)
#   --models_dir     Where to save models (default: ./models)
#   --sample_frac    Fraction of data to use (0.0-1.0)
#   --epochs         Training epochs (default: 30)
#   --skip_baseline  Skip baseline training (only train AD-RL-GNN)
#   --no_mlflow      Disable MLflow tracking
```

### Outputs

After training, the following artifacts are generated:

```
models/
├── fraudguard_AD_RL.pt      ← Production model
├── fraudguard_baseline.pt   ← Baseline for comparison
└── processed/
    ├── scaler.pkl           ← For API deployment
    └── pca.pkl              ← For API deployment

logs/mlruns/                 ← MLflow experiment tracking
```

## CI/CD

GitHub Actions workflow includes:

- **Linting**: black, isort, flake8
- **Testing**: pytest with coverage
- **Docker**: Build and push to registry

## License

MIT
