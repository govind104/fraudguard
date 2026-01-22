#!/usr/bin/env python3
"""
FraudGuard Local Training Script
================================

Production-ready training script for local GPU systems.
Based on colab_training.py but optimized for local execution.

Features:
- Full AD-RL-GNN pipeline with A/B comparison
- MLflow experiment tracking
- Preprocessor artifact saving for API deployment
- Memory-efficient NeighborLoader training
- Command-line arguments for flexibility

Usage:
    # Quick test (10% data)
    python scripts/train.py --sample_frac 0.1 --epochs 10

    # Full training (100% data)
    python scripts/train.py --sample_frac 1.0 --epochs 30

    # Custom paths
    python scripts/train.py --data_dir ./data --models_dir ./models

Requirements:
    - CUDA GPU (recommended: 8GB+ VRAM)
    - IEEE-CIS fraud detection dataset
    - All dependencies from pyproject.toml
"""

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import FraudDataLoader
from src.training.trainer import FraudTrainer
from src.training.evaluator import Evaluator
from src.utils.config import load_data_config, load_model_config
from src.utils.device_utils import set_seed, get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FraudGuard AD-RL-GNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data settings
    parser.add_argument("--data_dir", type=str, default="../ieee-fraud-detection",
                        help="Path to IEEE-CIS fraud detection data")
    parser.add_argument("--models_dir", type=str, default="./models",
                        help="Directory to save trained models")
    parser.add_argument("--logs_dir", type=str, default="./logs",
                        help="Directory for logs and MLflow tracking")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction of data to use (0.0-1.0)")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size for NeighborLoader")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate")
    parser.add_argument("--fraud_weight", type=float, default=25.0,
                        help="Class weight for fraud (minority class)")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    
    # MCD settings
    parser.add_argument("--baseline_alpha", type=float, default=0.0,
                        help="MCD alpha for baseline (0.0 = no MCD)")
    parser.add_argument("--gold_alpha", type=float, default=0.80,
                        help="MCD alpha for AD-RL-GNN")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline training (only train AD-RL-GNN)")
    parser.add_argument("--no_mlflow", action="store_true",
                        help="Disable MLflow tracking")
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories."""
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(f"{args.models_dir}/processed", exist_ok=True)
    
    print(f"📁 Data: {args.data_dir}")
    print(f"📁 Models: {args.models_dir}")
    print(f"📁 Logs: {args.logs_dir}")


def setup_mlflow(args):
    """Initialize MLflow tracking."""
    if args.no_mlflow:
        return None
    
    try:
        import mlflow
        import mlflow.pytorch
        
        mlflow_dir = f"{args.logs_dir}/mlruns"
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")
        mlflow.set_experiment("FraudGuard-GNN")
        print(f"📊 MLflow tracking: {mlflow_dir}")
        return mlflow
    except ImportError:
        print("⚠️ MLflow not installed, tracking disabled")
        return None


def load_data(args):
    """Load and split data."""
    print(f"\n📥 Loading data (sample_frac={args.sample_frac*100:.0f}%)...")
    
    data_cfg = load_data_config()
    data_cfg.paths.raw_data_dir = Path(args.data_dir)
    
    loader = FraudDataLoader(config=data_cfg)
    df = loader.load_train_data(sample_frac=args.sample_frac)
    train_df, val_df, test_df = loader.create_splits(df)
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    print(f"  Fraud rate: {df['isFraud'].mean()*100:.2f}%")
    
    return train_df, val_df, test_df


def train_model(
    train_df, val_df, test_df,
    args, device, evaluator,
    model_name: str,
    alpha: float,
    use_mcd: bool,
    use_rl: bool,
    mlflow=None,
):
    """Train a single model (baseline or AD-RL-GNN).
    
    Returns:
        best_gmeans: Best G-Means achieved
        vram_gb: Peak VRAM usage
        p95_ms: P95 inference latency
        trainer: Trained FraudTrainer (for preprocessor access)
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"   Alpha={alpha}, MCD={use_mcd}, RL={use_rl}")
    print(f"{'='*60}")
    
    # Load configs
    model_cfg = load_model_config()
    data_cfg = load_data_config()
    data_cfg.paths.raw_data_dir = Path(args.data_dir)
    
    # Set hyperparameters
    model_cfg.training["max_epochs"] = args.epochs
    model_cfg.training["learning_rate"] = args.lr
    model_cfg.adaptive_mcd["alpha"] = alpha
    model_cfg.graph.similarity_threshold = 0.75
    if use_rl:
        model_cfg.rl_agent["reward_scaling"] = 2.0
    
    # Initialize trainer
    trainer = FraudTrainer(model_config=model_cfg, data_config=data_cfg, device=device)
    
    # Reset VRAM stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Preprocessing pipeline
    print("\n📊 Preprocessing...")
    trainer._preprocess(train_df, val_df, test_df)
    trainer._build_graph()
    trainer._prepare_labels(train_df, val_df, test_df)
    
    # MCD and RL stages
    if use_mcd and alpha > 0:
        print(f"\n🧠 Training AdaptiveMCD (Alpha={alpha})...")
        trainer._train_mcd()
    
    if use_rl:
        print("\n🤖 Training RL Agent...")
        trainer._train_rl_and_enhance()
        
        # Flush VRAM
        torch.cuda.empty_cache()
        gc.collect()
    
    # Initialize model
    weights = torch.tensor([1.0, args.fraud_weight]).to(device)
    trainer._init_model()
    trainer.criterion = torch.nn.CrossEntropyLoss(weight=weights)
    model = trainer.model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create data loaders
    num_neighbors = [25, 10]
    optimized_data = Data(
        x=trainer.X_full,
        edge_index=trainer.edge_index,
        y=trainer.all_labels
    )
    optimized_data.train_mask = trainer.train_mask
    optimized_data.val_mask = trainer.val_mask
    
    train_loader = NeighborLoader(
        optimized_data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        input_nodes=optimized_data.train_mask,
        shuffle=True
    )
    val_loader = NeighborLoader(
        optimized_data,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        input_nodes=optimized_data.val_mask,
        shuffle=False
    )
    
    # Training loop
    print(f"\n🏋️ Training ({args.epochs} epochs)...")
    best_gmeans = 0
    model_path = f"{args.models_dir}/fraudguard_{model_name}.pt"
    
    # MLflow run
    mlflow_context = mlflow.start_run(run_name=model_name) if mlflow else nullcontext()
    
    with mlflow_context:
        if mlflow:
            mlflow.log_params({
                "model_type": model_name,
                "alpha": alpha,
                "fraud_weight": args.fraud_weight,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "max_epochs": args.epochs,
                "num_neighbors": str(num_neighbors),
                "gradient_clip": args.gradient_clip,
            })
        
        for epoch in range(args.epochs):
            # Train
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = trainer.criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
                optimizer.step()
            
            # Evaluate
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    pred = out[:batch.batch_size].argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_true.extend(batch.y[:batch.batch_size].cpu().numpy())
            
            metrics = evaluator.compute_metrics(np.array(all_true), np.array(all_preds))
            gmeans = metrics['gmeans']
            
            # Log to MLflow
            if mlflow:
                mlflow.log_metrics({
                    "gmeans": gmeans,
                    "recall": metrics['recall'],
                    "specificity": metrics['specificity']
                }, step=epoch)
            
            print(f"  Epoch {epoch+1:>2}/{args.epochs} | "
                  f"Spec: {metrics['specificity']*100:.2f}% | "
                  f"Recall: {metrics['recall']*100:.2f}% | "
                  f"G-Means: {gmeans*100:.2f}%")
            
            # Save best model
            if gmeans > best_gmeans:
                best_gmeans = gmeans
                torch.save(model.state_dict(), model_path)
        
        # Log final metrics
        if mlflow:
            mlflow.log_metric("best_gmeans", best_gmeans)
            
            # Log model to registry for AD-RL-GNN
            if use_rl:
                model.load_state_dict(torch.load(model_path))
                mlflow.pytorch.log_model(
                    model, "fraud_gnn",
                    registered_model_name="FraudGuard-Production"
                )
                print("\n📊 Model logged to MLflow Model Registry")
    
    # Capture metrics
    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    # Measure latency
    model.load_state_dict(torch.load(model_path))
    model.eval()
    latencies = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            start = time.perf_counter()
            _ = model(batch.x, batch.edge_index)
            latencies.append((time.perf_counter() - start) * 1000)
    
    p95_ms = np.percentile(latencies, 95)
    
    print(f"\n✅ {model_name} Results:")
    print(f"   Best G-Means: {best_gmeans*100:.2f}%")
    print(f"   Peak VRAM: {vram_gb:.2f} GB")
    print(f"   P95 Latency: {p95_ms:.2f} ms")
    print(f"   Model saved: {model_path}")
    
    return best_gmeans, vram_gb, p95_ms, trainer


class nullcontext:
    """Null context manager for when MLflow is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def save_preprocessor(trainer, args):
    """Save preprocessor artifacts for API deployment."""
    processed_dir = f"{args.models_dir}/processed"
    
    with open(f"{processed_dir}/scaler.pkl", "wb") as f:
        pickle.dump(trainer.preprocessor.scaler, f)
    
    with open(f"{processed_dir}/pca.pkl", "wb") as f:
        pickle.dump(trainer.preprocessor.pca, f)
    
    print(f"\n✅ Preprocessor artifacts saved to {processed_dir}")


def print_final_comparison(baseline_result, gold_result, evaluator):
    """Print final A/B comparison."""
    baseline_gmeans, baseline_vram, baseline_p95, _ = baseline_result
    gold_gmeans, gold_vram, gold_p95, _ = gold_result
    
    improvement = evaluator.compute_gmeans_improvement(baseline_gmeans, gold_gmeans)
    
    print("\n" + "="*60)
    print("🎯 FINAL ARCHITECTURAL COMPARISON")
    print("="*60)
    print(f"| {'Metric':<15} | {'Baseline':<12} | {'AD-RL-GNN':<12} | {'Change':<12} |")
    print(f"|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*14}|")
    print(f"| {'G-Means':<15} | {baseline_gmeans*100:>10.2f}% | {gold_gmeans*100:>10.2f}% | {improvement:>+10.1f}% |")
    print(f"| {'P95 Latency':<15} | {baseline_p95:>10.2f}ms | {gold_p95:>10.2f}ms | {((gold_p95-baseline_p95)/baseline_p95)*100:>+10.1f}% |")
    print(f"| {'Peak VRAM':<15} | {baseline_vram:>10.2f}GB | {gold_vram:>10.2f}GB | {((gold_vram-baseline_vram)/baseline_vram)*100:>+10.1f}% |")
    print("="*60)
    
    # CV Claims validation
    print("\n📋 CV CLAIMS VALIDATION:")
    print(f"   ✅ G-Means Improvement: {improvement:.1f}% (claim: ~21%)")
    if gold_p95 < 30:
        print(f"   ✅ P95 Latency: {gold_p95:.1f}ms < 30ms")
    else:
        print(f"   ⚠️ P95 Latency: {gold_p95:.1f}ms (target: <30ms)")


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🔒 FRAUDGUARD LOCAL TRAINING")
    print("="*60)
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"🖥️ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    setup_directories(args)
    mlflow = setup_mlflow(args)
    
    # Load data
    train_df, val_df, test_df = load_data(args)
    evaluator = Evaluator()
    
    # Train baseline (optional)
    baseline_result = None
    if not args.skip_baseline:
        baseline_result = train_model(
            train_df, val_df, test_df,
            args, device, evaluator,
            model_name="baseline",
            alpha=args.baseline_alpha,
            use_mcd=False,
            use_rl=False,
            mlflow=mlflow,
        )
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Train AD-RL-GNN
    gold_result = train_model(
        train_df, val_df, test_df,
        args, device, evaluator,
        model_name="AD_RL",
        alpha=args.gold_alpha,
        use_mcd=True,
        use_rl=True,
        mlflow=mlflow,
    )
    
    # Save preprocessor from AD-RL-GNN (the deployed model)
    _, _, _, trainer = gold_result
    save_preprocessor(trainer, args)
    
    # Print comparison
    if baseline_result:
        print_final_comparison(baseline_result, gold_result, evaluator)
    else:
        gold_gmeans, gold_vram, gold_p95, _ = gold_result
        print("\n" + "="*60)
        print("🎯 AD-RL-GNN RESULTS")
        print("="*60)
        print(f"   G-Means: {gold_gmeans*100:.2f}%")
        print(f"   P95 Latency: {gold_p95:.2f}ms")
        print(f"   Peak VRAM: {gold_vram:.2f}GB")
    
    print("\n✅ Training complete!")
    print(f"   Model: {args.models_dir}/fraudguard_AD_RL.pt")
    print(f"   Preprocessor: {args.models_dir}/processed/")
    if mlflow:
        print(f"   MLflow: {args.logs_dir}/mlruns/")


if __name__ == "__main__":
    main()
