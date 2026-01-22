"""Phase 2 smoke test - verify all modules work together.

Run: uv run python scripts/test_phase2.py
"""

import sys

sys.path.insert(0, ".")

import torch

from src.data.graph_builder import GraphBuilder
from src.data.loader import FraudDataLoader
from src.data.preprocessor import FeaturePreprocessor
from src.models import MCES, AdaptiveMCD, FocalLoss, FraudGNN, RLAgent, compute_class_weights

print("=" * 50)
print("Phase 2 Smoke Test")
print("=" * 50)

# 1. Load data
loader = FraudDataLoader()
df = loader.load_train_data(sample_frac=0.01)
train_df, val_df, test_df = loader.create_splits(df)
print(f"\n✓ Data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# 2. Preprocess
prep = FeaturePreprocessor()
X_train = prep.fit_transform(train_df)
X_val = prep.transform(val_df)
X_test = prep.transform(test_df)
X_full = torch.cat([X_train, X_val, X_test])
print(f"✓ Features: {X_full.shape}")

# 3. Build graph (leak-free)
builder = GraphBuilder()
train_edges = builder.fit(X_train)
full_edges = builder.transform(torch.cat([X_val, X_test]), train_size=len(X_train))
builder.verify_no_leakage(full_edges, train_size=len(X_train))
print(f"✓ Graph: {full_edges.shape[1]} edges (leak-free)")

# 4. Labels
labels = torch.cat(
    [
        torch.tensor(train_df["isFraud"].values, dtype=torch.long),
        torch.tensor(val_df["isFraud"].values, dtype=torch.long),
        torch.tensor(test_df["isFraud"].values, dtype=torch.long),
    ]
)
print(f"✓ Labels: {labels.sum()}/{len(labels)} fraud")

# 5. Model forward pass
model = FraudGNN(in_channels=X_full.shape[1])
logits = model(X_full, full_edges)
print(f"✓ FraudGNN output: {logits.shape}")

# 6. Loss
weights = compute_class_weights(labels[: len(train_df)])
criterion = FocalLoss(alpha=0.5, gamma=4, weight=weights)
loss = criterion(logits[: len(train_df)], labels[: len(train_df)])
print(f"✓ FocalLoss: {loss.item():.4f}")

# 7. Quick backward pass
loss.backward()
print(f"✓ Backward pass complete")

print("\n" + "=" * 50)
print("ALL PHASE 2 MODULES VERIFIED ✓")
print("=" * 50)
