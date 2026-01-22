"""Debug script to diagnose minority class collapse."""

import gc
import sys

sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.data.graph_builder import GraphBuilder
from src.data.loader import FraudDataLoader
from src.data.preprocessor import FeaturePreprocessor
from src.models import FocalLoss, FraudGNN, compute_class_weights
from src.utils.config import load_model_config
from src.utils.device_utils import get_device, set_seed

set_seed(42)
device = get_device()

print("Loading 10% sample data...")
loader = FraudDataLoader()
df = loader.load_train_data(sample_frac=0.1)
train_df, val_df, test_df = loader.create_splits(df)

print("Preprocessing...")
prep = FeaturePreprocessor()
X_train = prep.fit_transform(train_df)
X_val = prep.transform(val_df)
X_test = prep.transform(test_df)

n_train = len(X_train)
n_val = len(X_val)
n_test = len(X_test)

train_labels = torch.tensor(train_df["isFraud"].values, dtype=torch.long)
val_labels = torch.tensor(val_df["isFraud"].values, dtype=torch.long)
test_labels = torch.tensor(test_df["isFraud"].values, dtype=torch.long)

X_full = torch.cat([X_train, X_val, X_test])
all_labels = torch.cat([train_labels, val_labels, test_labels])

del df, train_df, val_df, test_df
gc.collect()

print("Loading cached graph...")
graph_cache = Path("data/graphs/edges_local.pt")
edge_index = torch.load(graph_cache)

X_full = X_full.to(device)
edge_index = edge_index.to(device)
all_labels = all_labels.to(device)

train_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool, device=device)
val_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool, device=device)
train_mask[:n_train] = True
val_mask[n_train : n_train + n_val] = True

data = Data(x=X_full, edge_index=edge_index, y=all_labels)
data.train_mask = train_mask
data.val_mask = val_mask

train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=4096,
    input_nodes=train_mask,
    shuffle=True,
)

print("Loading model...")
model = FraudGNN(in_channels=X_full.shape[1]).to(device)
model.load_state_dict(torch.load("models/best_model.pt"))

# ============== DEBUG CELL: Minority Class Collapse Diagnosis ==============
print("\n" + "=" * 60)
print("DEBUG: Minority Class Collapse Diagnosis")
print("=" * 60)

# 1. Class Weights
print("\n[1] CLASS WEIGHTS:")
weights = compute_class_weights(train_labels, device)
print(f"  Raw tensor: {weights}")
print(f"  Class 0 (Legit) weight: {weights[0].item():.4f}")
print(f"  Class 1 (Fraud) weight: {weights[1].item():.4f}")
print(f"  Ratio (Fraud/Legit): {weights[1].item() / weights[0].item():.2f}x")

# 2. Mini-Batch Label Distribution
print("\n[2] MINI-BATCH LABEL DISTRIBUTION:")
batch = next(iter(train_loader))
batch = batch.to(device)
batch_labels = batch.y[: batch.batch_size]
fraud_count = (batch_labels == 1).sum().item()
legit_count = (batch_labels == 0).sum().item()
total = len(batch_labels)
print(f"  Batch size (target nodes): {total}")
print(f"  Legit (0): {legit_count} ({legit_count/total*100:.1f}%)")
print(f"  Fraud (1): {fraud_count} ({fraud_count/total*100:.1f}%)")
print(f"  Ratio: {fraud_count/max(legit_count,1):.4f}")

# 3. Raw Model Outputs (Logits)
print("\n[3] RAW MODEL OUTPUTS:")
model.eval()
with torch.no_grad():
    logits = model(batch.x, batch.edge_index)[: batch.batch_size]
    probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1)

print(f"  Logits shape: {logits.shape}")
print(f"  Logits (Class 0) - Mean: {logits[:, 0].mean():.4f}, Std: {logits[:, 0].std():.4f}")
print(f"  Logits (Class 1) - Mean: {logits[:, 1].mean():.4f}, Std: {logits[:, 1].std():.4f}")
print(f"  Logit Difference (C1-C0) Mean: {(logits[:, 1] - logits[:, 0]).mean():.4f}")

print("\n[4] PREDICTION DISTRIBUTION:")
pred_fraud = (pred_class == 1).sum().item()
pred_legit = (pred_class == 0).sum().item()
print(f"  Predicted Legit: {pred_legit} ({pred_legit/total*100:.1f}%)")
print(f"  Predicted Fraud: {pred_fraud} ({pred_fraud/total*100:.1f}%)")

print("\n[5] PROBABILITY HISTOGRAM (Class 1 / Fraud):")
fraud_probs = probs[:, 1].cpu().numpy()
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(fraud_probs, bins=bins)
for i, count in enumerate(hist):
    bar = "█" * int(count / max(hist) * 20) if max(hist) > 0 else ""
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {count:4d} {bar}")

print("\n" + "=" * 60)
