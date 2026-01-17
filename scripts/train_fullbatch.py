"""Full-batch training script (no NeighborLoader dependency).

This script avoids NeighborLoader which requires torch-sparse/pyg-lib
that can have Windows DLL loading issues. It uses standard PyTorch
batching instead.

Run: uv run python scripts/train_fullbatch.py --sample_frac 0.3 --epochs 100
"""
import sys
import argparse
import gc
from datetime import datetime
from pathlib import Path
sys.path.insert(0, ".")

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import time

from src.data.loader import FraudDataLoader
from src.data.preprocessor import FeaturePreprocessor  
from src.data.graph_builder import GraphBuilder
from src.models import FraudGNN, FocalLoss, compute_class_weights
from src.utils.device_utils import set_seed, get_device
from src.utils.config import load_model_config

# CV Claims
CV_CLAIMS = {
    "specificity": 98.72,
    "gmeans_improvement": 18.11,
    "p95_latency_ms": 100,
}


def main():
    parser = argparse.ArgumentParser(description="Train FraudGuard (Full-Batch)")
    parser.add_argument("--sample_frac", type=float, default=0.3, 
                        help="Data fraction (use 0.3 for local, 1.0 for full)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print("FRAUDGUARD FULL-BATCH TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sample: {args.sample_frac*100:.0f}%")
    print(f"Threshold: {args.threshold}")
    
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # ===== PHASE 1: Load Data =====
    print("\n[1/5] Loading data...")
    loader = FraudDataLoader()
    df = loader.load_train_data(sample_frac=args.sample_frac)
    train_df, val_df, test_df = loader.create_splits(df)
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    print(f"  Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    
    # ===== PHASE 2: Preprocess =====
    print("\n[2/5] Preprocessing features...")
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)
    X_test = prep.transform(test_df)
    
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    
    # Extract labels
    train_labels = torch.tensor(train_df["isFraud"].values, dtype=torch.long)
    val_labels = torch.tensor(val_df["isFraud"].values, dtype=torch.long)
    test_labels = torch.tensor(test_df["isFraud"].values, dtype=torch.long)
    
    # Concatenate
    X_full = torch.cat([X_train, X_val, X_test])
    all_labels = torch.cat([train_labels, val_labels, test_labels])
    
    print(f"  Features shape: {X_full.shape}")
    
    # Cleanup
    del df, train_df, val_df, test_df
    gc.collect()
    
    # ===== PHASE 3: Build Graph =====
    cache_file = f"data/graphs/edges_sample_{args.sample_frac}.pt"
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        print(f"\n[3/5] Loading cached graph from {cache_file}...")
        edge_index = torch.load(cache_path)
        print(f"  Loaded {edge_index.shape[1]:,} edges")
    else:
        print(f"\n[3/5] Building graph (threshold={args.threshold})...")
        
        model_cfg = load_model_config()
        model_cfg.graph.similarity_threshold = args.threshold
        model_cfg.graph.max_neighbors = 50
        model_cfg.graph.batch_size = 50000
        
        builder = GraphBuilder(config=model_cfg)
        train_edges = builder.fit(X_train)
        edge_index = builder.transform(torch.cat([X_val, X_test]), train_size=n_train)
        builder.verify_no_leakage(edge_index, train_size=n_train)
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index, cache_path)
        print(f"  Cached to {cache_file}")
    
    print(f"  Total edges: {edge_index.shape[1]:,}")
    
    # Cleanup
    del X_train, X_val, X_test
    gc.collect()
    
    # Move to device
    X_full = X_full.to(device)
    edge_index = edge_index.to(device)
    all_labels = all_labels.to(device)
    
    # Create masks
    train_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    # ===== PHASE 4: Train (Full-Batch) =====
    print("\n[4/5] Training model (full-batch)...")
    
    model = FraudGNN(in_channels=X_full.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    weights = compute_class_weights(train_labels, device)
    criterion = FocalLoss(alpha=0.5, gamma=4, weight=weights)
    
    best_gmeans = 0
    patience = 30
    patience_counter = 0
    
    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'Spec':>8} | {'Recall':>8} | {'F1':>8} | {'G-Means':>8}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training (full batch on training nodes)
        model.train()
        optimizer.zero_grad()
        out = model(X_full, edge_index)
        loss = criterion(out[train_mask], all_labels[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(X_full, edge_index)
                val_pred = out[val_mask].argmax(dim=1)
                val_true = all_labels[val_mask]
            
            cm = confusion_matrix(val_true.cpu(), val_pred.cpu(), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmeans = np.sqrt(tpr * tnr)
            f1 = f1_score(val_true.cpu(), val_pred.cpu(), zero_division=0)
            
            print(f"{epoch+1:>5} | {loss.item():>8.4f} | {tnr*100:>7.2f}% | {tpr*100:>7.2f}% | {f1*100:>7.2f}% | {gmeans*100:>7.2f}%")
            
            if gmeans > best_gmeans:
                best_gmeans = gmeans
                patience_counter = 0
                torch.save(model.state_dict(), "models/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience // 5:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
    
    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time/60:.1f} minutes")
    print(f"Best validation G-Means: {best_gmeans*100:.2f}%")
    
    # ===== PHASE 5: Test Evaluation =====
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    
    latencies = []
    with torch.no_grad():
        for _ in range(10):  # Multiple runs for latency
            start = time.perf_counter()
            out = model(X_full, edge_index)
            latencies.append((time.perf_counter() - start) * 1000)
        
        test_pred = out[test_mask].argmax(dim=1)
        test_true = all_labels[test_mask]
    
    cm = confusion_matrix(test_true.cpu(), test_pred.cpu(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmeans = np.sqrt(tpr * tnr)
    f1 = f1_score(test_true.cpu(), test_pred.cpu(), zero_division=0)
    
    print(f"\nConfusion Matrix: TP={tp:,}, TN={tn:,}, FP={fp:,}, FN={fn:,}")
    print(f"\nSpecificity:  {tnr*100:.2f}%  (CV target: {CV_CLAIMS['specificity']}%)")
    print(f"Recall:       {tpr*100:.2f}%")
    print(f"F1 Score:     {f1*100:.2f}%")
    print(f"G-Means:      {gmeans*100:.2f}%")
    print(f"\nLatency P95:  {np.percentile(latencies, 95):.1f}ms  (CV target: <{CV_CLAIMS['p95_latency_ms']}ms)")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save({
        "model_state_dict": model.state_dict(),
        "specificity": tnr,
        "gmeans": gmeans,
        "f1": f1,
    }, f"models/fraudguard_{timestamp}.pt")
    
    print(f"\nModel saved: models/fraudguard_{timestamp}.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
