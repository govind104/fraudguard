# FraudGuard Training Notebook for Google Colab

This notebook trains the FraudGuard AD-RL-GNN model on the full IEEE-CIS dataset using Colab GPU resources.

## Setup

```python
# @title 1. Setup Environment
# Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/govind104/fraudguard.git
%cd fraudguard

# Install dependencies
!pip install -q torch torch-geometric faiss-gpu pandas numpy scikit-learn pyyaml structlog

# Install repo in editable mode
!pip install -e .

print("✓ Environment setup complete")
```

## Configuration

```python
# @title 2. Configuration
import os

# Data paths - UPDATE THESE to your Google Drive paths
DATA_DIR = "/content/drive/MyDrive/ieee-fraud-detection"
MODELS_DIR = "/content/drive/MyDrive/fraudguard-models"
LOGS_DIR = "/content/drive/MyDrive/fraudguard-logs"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Training parameters
SAMPLE_FRAC = 1.0  # Use full dataset
MAX_EPOCHS = 100
BATCH_SIZE = 4096  # For NeighborLoader

print(f"Data: {DATA_DIR}")
print(f"Models: {MODELS_DIR}")
print(f"Logs: {LOGS_DIR}")
```

## Check GPU

```python
# @title 3. Verify GPU and FAISS
import torch
import faiss

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nFAISS GPUs: {faiss.get_num_gpus()}")
```

## Load Data

```python
# @title 4. Load and Preprocess Data
import sys
sys.path.insert(0, '/content/fraudguard')

from src.data.loader import FraudDataLoader
from src.data.preprocessor import FeaturePreprocessor
from src.data.graph_builder import GraphBuilder
from src.utils.device_utils import set_seed, get_device

set_seed(42)
device = get_device()
print(f"Using device: {device}")

# Load data
loader = FraudDataLoader()
df = loader.load_train_data(sample_frac=SAMPLE_FRAC)
train_df, val_df, test_df = loader.create_splits(df)

print(f"\nData loaded:")
print(f"  Train: {len(train_df):,}")
print(f"  Val: {len(val_df):,}")
print(f"  Test: {len(test_df):,}")
print(f"  Fraud rate: {df['isFraud'].mean()*100:.2f}%")
```

## Build Graph (or Load Cached)

```python
# @title 5. Build or Load Graph
import torch
import os

GRAPH_CACHE = f"{MODELS_DIR}/edges_full.pt"
FEATURES_CACHE = f"{MODELS_DIR}/features_full.pt"

# Preprocess
prep = FeaturePreprocessor()
X_train = prep.fit_transform(train_df).to(device)
X_val = prep.transform(val_df).to(device)
X_test = prep.transform(test_df).to(device)
X_full = torch.cat([X_train, X_val, X_test])

print(f"Features shape: {X_full.shape}")

# Check for cached graph
if os.path.exists(GRAPH_CACHE):
    print("Loading cached graph...")
    edge_index = torch.load(GRAPH_CACHE)
    print(f"Loaded {edge_index.shape[1]:,} edges")
else:
    print("Building graph (this may take 30-60 minutes)...")
    builder = GraphBuilder()
    train_edges = builder.fit(X_train)
    edge_index = builder.transform(torch.cat([X_val, X_test]), train_size=len(X_train))
    
    # Verify leak-free
    builder.verify_no_leakage(edge_index, train_size=len(X_train))
    
    # Cache to Drive
    torch.save(edge_index, GRAPH_CACHE)
    torch.save(X_full, FEATURES_CACHE)
    print(f"Cached graph to {GRAPH_CACHE}")

edge_index = edge_index.to(device)
print(f"Graph: {edge_index.shape[1]:,} edges")
```

## Mini-Batch Training with NeighborLoader

```python
# @title 6. Setup Mini-Batch Training
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import torch.nn.functional as F

# Prepare labels
train_labels = torch.tensor(train_df["isFraud"].values, dtype=torch.long)
val_labels = torch.tensor(val_df["isFraud"].values, dtype=torch.long)
test_labels = torch.tensor(test_df["isFraud"].values, dtype=torch.long)
all_labels = torch.cat([train_labels, val_labels, test_labels]).to(device)

# Masks
n = len(X_full)
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask = torch.zeros(n, dtype=torch.bool)
test_mask = torch.zeros(n, dtype=torch.bool)
train_mask[:len(X_train)] = True
val_mask[len(X_train):len(X_train)+len(X_val)] = True
test_mask[len(X_train)+len(X_val):] = True

# Create PyG Data object
data = Data(x=X_full, edge_index=edge_index, y=all_labels)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Create NeighborLoader for mini-batch training
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 2-hop neighborhood
    batch_size=BATCH_SIZE,
    input_nodes=train_mask,
    shuffle=True,
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=BATCH_SIZE,
    input_nodes=val_mask,
    shuffle=False,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

## Train Model

```python
# @title 7. Train Model
from src.models import FraudGNN, FocalLoss, compute_class_weights
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

# Model
model = FraudGNN(in_channels=X_full.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
weights = compute_class_weights(train_labels, device)
criterion = FocalLoss(alpha=0.5, gamma=4, weight=weights)

# Training loop
best_gmeans = 0
patience = 30
patience_counter = 0

for epoch in range(MAX_EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # Validation every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                pred = out[:batch.batch_size].argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(batch.y[:batch.batch_size].cpu().numpy())
        
        cm = confusion_matrix(all_true, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmeans = np.sqrt(tpr * tnr)
        f1 = f1_score(all_true, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
              f"Spec: {tnr*100:.2f}% | Recall: {tpr*100:.2f}% | "
              f"F1: {f1*100:.2f}% | G-Means: {gmeans*100:.2f}%")
        
        # Early stopping
        if gmeans > best_gmeans:
            best_gmeans = gmeans
            patience_counter = 0
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience // 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

print(f"\nBest validation G-Means: {best_gmeans*100:.2f}%")
```

## Evaluate on Test Set

```python
# @title 8. Test Evaluation
import time

# Load best model
model.load_state_dict(torch.load(f"{MODELS_DIR}/best_model.pt"))
model.eval()

# Full graph evaluation
test_loader = NeighborLoader(
    data,
    num_neighbors=[-1, -1],  # Full neighborhood
    batch_size=BATCH_SIZE,
    input_nodes=test_mask,
    shuffle=False,
)

all_preds, all_true = [], []
latencies = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        start = time.perf_counter()
        out = model(batch.x, batch.edge_index)
        latencies.append((time.perf_counter() - start) * 1000)
        pred = out[:batch.batch_size].argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_true.extend(batch.y[:batch.batch_size].cpu().numpy())

# Metrics
cm = confusion_matrix(all_true, all_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
gmeans = np.sqrt(tpr * tnr)
f1 = f1_score(all_true, all_preds, zero_division=0)

print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"\nSpecificity:  {tnr*100:.2f}%  (CV target: 98.72%)")
print(f"Recall:       {tpr*100:.2f}%")
print(f"F1 Score:     {f1*100:.2f}%")
print(f"G-Means:      {gmeans*100:.2f}%")
print(f"\nLatency Mean: {np.mean(latencies):.1f}ms")
print(f"Latency P95:  {np.percentile(latencies, 95):.1f}ms  (CV target: <100ms)")
```

## Save Final Model

```python
# @title 9. Save Model
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "in_channels": X_full.shape[1],
        "specificity": tnr,
        "gmeans": gmeans,
        "f1": f1,
    }
}, f"{MODELS_DIR}/fraudguard_final.pt")

print(f"Model saved to {MODELS_DIR}/fraudguard_final.pt")
```
