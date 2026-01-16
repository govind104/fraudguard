"""Full pipeline training script.

Trains FraudGuard on full dataset with all components:
- AdaptiveMCD downsampling
- RL-driven MCES subgraph selection
- GNN with FocalLoss
- Early stopping and checkpointing

Run: uv run python scripts/train_full.py [--sample_frac 1.0] [--epochs 100]
"""
import sys
import argparse
sys.path.insert(0, ".")

from src.data.loader import FraudDataLoader
from src.training import FraudTrainer, EarlyStopping, ModelCheckpoint, MetricsLogger, Evaluator
from src.utils.device_utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Train FraudGuard")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Data sample fraction")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_mcd", action="store_true", help="Disable AdaptiveMCD")
    parser.add_argument("--no_rl", action="store_true", help="Disable RL subgraph selection")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FRAUDGUARD FULL PIPELINE TRAINING")
    print("=" * 60)
    print(f"Sample fraction: {args.sample_frac}")
    print(f"Max epochs: {args.epochs}")
    print(f"AdaptiveMCD: {'ON' if not args.no_mcd else 'OFF'}")
    print(f"RL Subgraph: {'ON' if not args.no_rl else 'OFF'}")
    print("=" * 60 + "\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    loader = FraudDataLoader()
    df = loader.load_train_data(sample_frac=args.sample_frac)
    train_df, val_df, test_df = loader.create_splits(df)
    
    print(f"\nData: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%\n")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor="val_gmeans", patience=30, mode="max"),
        ModelCheckpoint(path="models", monitor="val_gmeans", mode="max"),
        MetricsLogger(log_dir="logs"),
    ]
    
    # Train
    trainer = FraudTrainer(callbacks=callbacks)
    metrics = trainer.fit(
        train_df, val_df, test_df,
        max_epochs=args.epochs,
        use_mcd=not args.no_mcd,
        use_rl=not args.no_rl,
    )
    
    # Benchmark latency
    latency = trainer.benchmark_latency(n_runs=100)
    
    # Save model
    trainer.save("models/fraudguard_final.pt")
    
    # Print results
    evaluator = Evaluator()
    evaluator.print_report(metrics, title="FINAL TEST RESULTS")
    
    print(f"\nInference Latency:")
    print(f"  Mean: {latency['mean_ms']:.1f}ms")
    print(f"  P95:  {latency['p95_ms']:.1f}ms")
    print(f"  P99:  {latency['p99_ms']:.1f}ms")
    
    print(f"\nTraining time: {metrics.get('train_time', 0):.1f}s")
    print(f"Model saved to: models/fraudguard_final.pt")


if __name__ == "__main__":
    main()
