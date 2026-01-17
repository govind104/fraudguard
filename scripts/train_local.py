"""Memory-optimized local training script.

Applies all OOM fixes from Colab:
- Chunked graph building (50k nodes at a time)
- Directed graph (no symmetrization)
- Aggressive memory cleanup
- 0.9 threshold for fewer edges

Run: uv run python scripts/train_local.py [--sample_frac 1.0] [--epochs 100]
"""
import sys
import argparse
import gc
from datetime import datetime
from pathlib import Path
sys.path.insert(0, ".")

import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, f1_score

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
    parser = argparse.ArgumentParser(description="Train FraudGuard (Memory-Optimized)")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9, help="Graph similarity threshold")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print("FRAUDGUARD LOCAL TRAINING (Memory-Optimized)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sample: {args.sample_frac*100:.0f}%")
    print(f"Threshold: {args.threshold}")
    print(f"Batch size: {args.batch_size}")
    
    # Create directories
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
    
    # ===== PHASE 2: Preprocess & Extract Labels =====
    print("\n[2/5] Preprocessing features (CPU)...")
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)
    X_test = prep.transform(test_df)
    
    # Save lengths before deleting
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    
    # Extract labels before deleting dataframes
    train_labels = torch.tensor(train_df["isFraud"].values, dtype=torch.long)
    val_labels = torch.tensor(val_df["isFraud"].values, dtype=torch.long)
    test_labels = torch.tensor(test_df["isFraud"].values, dtype=torch.long)
    
    # Concatenate features
    X_full = torch.cat([X_train, X_val, X_test])
    print(f"  Features shape: {X_full.shape}")
    
    # Aggressive cleanup
    del df, train_df, val_df, test_df
    gc.collect()
    
    # ===== PHASE 3: Build Graph =====
    graph_cache = Path("data/graphs/edges_local.pt")
    
    if graph_cache.exists():
        print(f"\n[3/5] Loading cached graph from {graph_cache}...")
        edge_index = torch.load(graph_cache)
        print(f"  Loaded {edge_index.shape[1]:,} edges")
    else:
        print(f"\n[3/5] Building graph (threshold={args.threshold})...")
        
        # Override config with CLI threshold
        model_cfg = load_model_config()
        model_cfg.graph.similarity_threshold = args.threshold
        model_cfg.graph.max_neighbors = 50
        model_cfg.graph.batch_size = 50000
        
        builder = GraphBuilder(config=model_cfg)
        
        # Build train-train edges
        print("  Phase 1: Train -> Train...")
        train_edges = builder.fit(X_train)
        
        # Build val/test -> train edges
        print("  Phase 2: Val/Test -> Train...")
        edge_index = builder.transform(torch.cat([X_val, X_test]), train_size=n_train)
        
        # Verify leak-free
        builder.verify_no_leakage(edge_index, train_size=n_train)
        
        # Cache for future runs
        graph_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index, graph_cache)
        print(f"  Cached to {graph_cache}")
    
    print(f"  Total edges: {edge_index.shape[1]:,}")
    
    # Cleanup
    del X_train, X_val, X_test
    gc.collect()
    
    # Move to device
    X_full = X_full.to(device)
    edge_index = edge_index.to(device)
    
    # ===== PHASE 4: Setup Mini-Batch Training =====
    print("\n[4/5] Setting up mini-batch training...")
    
    all_labels = torch.cat([train_labels, val_labels, test_labels]).to(device)
    
    # Create masks
    train_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool)
    val_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool)
    test_mask = torch.zeros(n_train + n_val + n_test, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    # PyG Data object
    data = Data(x=X_full, edge_index=edge_index, y=all_labels)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # NeighborLoaders (no num_workers for compatibility)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=args.batch_size,
        input_nodes=train_mask,
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=args.batch_size,
        input_nodes=val_mask,
        shuffle=False,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ===== PHASE 5: Train =====
    print("\n[5/5] Training model...")
    
    model = FraudGNN(in_channels=X_full.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    weights = compute_class_weights(train_labels, device)
    criterion = FocalLoss(alpha=0.5, gamma=4, weight=weights)
    
    best_gmeans = 0
    patience = 30
    patience_counter = 0
    
    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'Spec':>8} | {'Recall':>8} | {'F1':>8} | {'G-Means':>8}")
    print("-" * 60)
    
    import time
    start_time = time.time()
    
    for epoch in range(args.epochs):
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
            
            print(f"{epoch+1:>5} | {avg_loss:>8.4f} | {tnr*100:>7.2f}% | {tpr*100:>7.2f}% | {f1*100:>7.2f}% | {gmeans*100:>7.2f}%")
            
            # Early stopping
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
    
    # ===== Test Evaluation =====
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=args.batch_size,
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
    
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmeans = np.sqrt(tpr * tnr)
    f1 = f1_score(all_true, all_preds, zero_division=0)
    
    print(f"\nConfusion Matrix: TP={tp:,}, TN={tn:,}, FP={fp:,}, FN={fn:,}")
    print(f"\nSpecificity:  {tnr*100:.2f}%  (CV target: {CV_CLAIMS['specificity']}%)")
    print(f"Recall:       {tpr*100:.2f}%")
    print(f"F1 Score:     {f1*100:.2f}%")
    print(f"G-Means:      {gmeans*100:.2f}%")
    print(f"\nLatency P95:  {np.percentile(latencies, 95):.1f}ms  (CV target: <{CV_CLAIMS['p95_latency_ms']}ms)")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save({
        "model_state_dict": model.state_dict(),
        "specificity": tnr,
        "gmeans": gmeans,
        "f1": f1,
    }, f"models/fraudguard_local_{timestamp}.pt")
    
    print(f"\nModel saved: models/fraudguard_local_{timestamp}.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
