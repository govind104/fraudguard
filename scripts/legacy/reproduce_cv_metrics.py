"""Reproduce CV metrics and verify claims.

Trains with optimal settings and compares against CV claims:
- Specificity: 98.72%
- G-Means Improvement: 18.11%
- P95 Latency: <100ms

Run: uv run python scripts/reproduce_cv_metrics.py [--sample_frac 1.0]
"""

import argparse
import sys

sys.path.insert(0, ".")

from src.data.loader import FraudDataLoader
from src.training import EarlyStopping, Evaluator, FraudTrainer, MetricsLogger, ModelCheckpoint
from src.utils.device_utils import set_seed

# CV Claims from original coursework
CV_CLAIMS = {
    "specificity": 98.72,
    "gmeans_improvement": 18.11,
    "p95_latency_ms": 100,
}


def train_baseline(train_df, val_df, test_df, epochs=50):
    """Train baseline GNN without AdaptiveMCD or RL."""
    print("\n[Step 1/3] Training BASELINE (no MCD, no RL)...")
    trainer = FraudTrainer(callbacks=[EarlyStopping(patience=20)])
    metrics = trainer.fit(
        train_df,
        val_df,
        test_df,
        max_epochs=epochs,
        use_mcd=False,
        use_rl=False,
    )
    return metrics


def train_full(train_df, val_df, test_df, epochs=100):
    """Train full AD-RL-GNN pipeline."""
    print("\n[Step 2/3] Training FULL PIPELINE (MCD + RL)...")
    callbacks = [
        EarlyStopping(monitor="val_gmeans", patience=30, mode="max"),
        ModelCheckpoint(path="models", monitor="val_gmeans", mode="max"),
    ]
    trainer = FraudTrainer(callbacks=callbacks)
    metrics = trainer.fit(
        train_df,
        val_df,
        test_df,
        max_epochs=epochs,
        use_mcd=True,
        use_rl=True,
    )
    latency = trainer.benchmark_latency(n_runs=100)
    trainer.save("models/fraudguard_cv.pt")

    return metrics, latency, trainer


def main():
    parser = argparse.ArgumentParser(description="Reproduce CV metrics")
    parser.add_argument("--sample_frac", type=float, default=0.3, help="Data sample fraction")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("CV METRICS REPRODUCTION")
    print("=" * 60)
    print(f"Target: Specificity={CV_CLAIMS['specificity']:.2f}%, ", end="")
    print(f"G-Means Improvement={CV_CLAIMS['gmeans_improvement']:.2f}%")
    print("=" * 60)

    set_seed(args.seed)

    # Load data
    loader = FraudDataLoader()
    df = loader.load_train_data(sample_frac=args.sample_frac)
    train_df, val_df, test_df = loader.create_splits(df)

    print(f"\nData: {len(df)} samples ({args.sample_frac*100:.0f}% of full dataset)")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

    # Train baseline
    baseline_metrics = train_baseline(train_df, val_df, test_df, epochs=args.epochs // 2)

    # Train full pipeline
    full_metrics, latency, trainer = train_full(train_df, val_df, test_df, epochs=args.epochs)

    # Compute G-Means improvement
    evaluator = Evaluator()
    gmeans_improvement = evaluator.compute_gmeans_improvement(
        baseline_metrics["gmeans"],
        full_metrics["gmeans"],
    )

    # Prepare achieved metrics
    achieved = {
        "specificity": full_metrics["specificity"] * 100,
        "gmeans_improvement": gmeans_improvement,
        "p95_latency_ms": latency["p95_ms"],
    }

    # Print comparison
    print("\n[Step 3/3] RESULTS COMPARISON")
    evaluator.print_cv_comparison(achieved, CV_CLAIMS)

    # Summary
    print("\nDetailed Metrics:")
    print(f"  Baseline G-Means:  {baseline_metrics['gmeans']*100:.2f}%")
    print(f"  Full G-Means:      {full_metrics['gmeans']*100:.2f}%")
    print(f"  Improvement:       {gmeans_improvement:.2f}%")
    print(f"\n  Specificity:       {full_metrics['specificity']*100:.2f}%")
    print(f"  Recall:            {full_metrics['recall']*100:.2f}%")
    print(f"  F1 Score:          {full_metrics['f1']*100:.2f}%")
    print(f"\n  P95 Latency:       {latency['p95_ms']:.1f}ms")

    print("\nModel saved to: models/fraudguard_cv.pt")


if __name__ == "__main__":
    main()
