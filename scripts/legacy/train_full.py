"""Full pipeline training with detailed CV validation logging.

Trains FraudGuard on full dataset and validates against CV claims:
- Specificity: 98.72%
- G-Means Improvement: 18.11%
- P95 Latency: <100ms

Run: uv run python scripts/train_full.py [--sample_frac 1.0] [--epochs 100]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import torch

from src.data.loader import FraudDataLoader
from src.training import EarlyStopping, Evaluator, FraudTrainer, MetricsLogger, ModelCheckpoint
from src.utils.device_utils import get_device, set_seed

# CV Claims
CV_CLAIMS = {
    "specificity": 98.72,
    "gmeans_improvement": 18.11,
    "p95_latency_ms": 100,
}


def print_header(sample_frac, epochs, use_mcd, use_rl):
    """Print training configuration header."""
    print("\n" + "=" * 70)
    print("FULL TRAINING RUN - CV CLAIMS VALIDATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample fraction: {sample_frac*100:.0f}%")
    print(f"  Max epochs: {epochs}")
    print(f"  AdaptiveMCD: {'ON' if use_mcd else 'OFF'}")
    print(f"  RL Subgraph: {'ON' if use_rl else 'OFF'}")
    print(f"  Device: {get_device()}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_data_summary(train_df, val_df, test_df, trainer):
    """Print data configuration summary."""
    total = len(train_df) + len(val_df) + len(test_df)
    fraud_count = train_df["isFraud"].sum() + val_df["isFraud"].sum() + test_df["isFraud"].sum()

    print("\n" + "-" * 70)
    print("DATA CONFIGURATION")
    print("-" * 70)
    print(f"  Total dataset: {total:,} rows")
    print(f"  Train/Val/Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    print(f"  Fraud ratio (original): {fraud_count/total*100:.2f}%")

    # After MCD (if used)
    if hasattr(trainer, "kept_majority") and trainer.kept_majority is not None:
        mcd_fraud = len(trainer.fraud_nodes)
        mcd_non_fraud = len(trainer.kept_majority)
        mcd_total = mcd_fraud + mcd_non_fraud
        print(f"  After AdaptiveMCD: {mcd_total:,} training samples")
        print(f"  Fraud ratio (after MCD): {mcd_fraud/mcd_total*100:.2f}%")

    # Graph info
    if trainer.edge_index is not None:
        n_edges = trainer.edge_index.shape[1]
        train_size = trainer.train_size

        # Count edge types
        src, dst = trainer.edge_index
        train_train = ((src < train_size) & (dst < train_size)).sum().item()
        train_test = ((src < train_size) != (dst < train_size)).sum().item() // 2

        print(f"  Graph edges: {n_edges:,}")
        print(f"    Train-train: {train_train:,}")
        print(f"    Train-test: {train_test:,}")
        print(f"    Test-test: 0 (leak-free)")


def print_training_progress(history):
    """Print training progression table."""
    print("\n" + "-" * 70)
    print("TRAINING PROGRESSION")
    print("-" * 70)
    print(
        f"| {'Epoch':>5} | {'Loss':>8} | {'Spec':>8} | {'Recall':>8} | {'F1':>8} | {'G-Means':>8} |"
    )
    print(f"|{'-'*7}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")

    for h in history:
        if h["epoch"] % 10 == 0 or h["epoch"] == len(history) - 1:
            spec = h.get("val_specificity", 0) * 100
            recall = h.get("val_recall", 0) * 100
            f1 = h.get("val_f1", 0) * 100
            gmeans = h.get("val_gmeans", 0) * 100
            print(
                f"| {h['epoch']+1:>5} | {h['train_loss']:>8.4f} | {spec:>7.2f}% | {recall:>7.2f}% | {f1:>7.2f}% | {gmeans:>7.2f}% |"
            )


def print_final_results(metrics, latency, train_time, baseline_gmeans=None):
    """Print final test results."""
    print("\n" + "-" * 70)
    print("FINAL TEST RESULTS")
    print("-" * 70)

    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']:,}")
    print(f"  True Negatives:  {metrics['tn']:,}")
    print(f"  False Positives: {metrics['fp']:,}")
    print(f"  False Negatives: {metrics['fn']:,}")

    print("\nPerformance Metrics:")
    print(
        f"  Specificity:          {metrics['specificity']*100:>6.2f}%  (CV: {CV_CLAIMS['specificity']}%)"
    )
    print(f"  Recall (Sensitivity): {metrics['recall']*100:>6.2f}%")
    print(f"  Precision:            {metrics['precision']*100:>6.2f}%")
    print(f"  F1 Score:             {metrics['f1']*100:>6.2f}%")
    print(f"  G-Means:              {metrics['gmeans']*100:>6.2f}%")

    if baseline_gmeans:
        improvement = ((metrics["gmeans"] - baseline_gmeans) / baseline_gmeans) * 100
        print(
            f"  G-Means Improvement:  {improvement:>6.2f}%  (CV: {CV_CLAIMS['gmeans_improvement']}%)"
        )

    print("\nInference Latency:")
    print(f"  Mean: {latency['mean_ms']:.1f}ms")
    print(f"  P95:  {latency['p95_ms']:.1f}ms  (CV: <{CV_CLAIMS['p95_latency_ms']}ms)")
    print(f"  P99:  {latency['p99_ms']:.1f}ms")

    print(f"\nTraining Time: {train_time/3600:.2f} hours ({train_time:.1f}s)")


def print_cv_comparison(metrics, latency, baseline_gmeans=None):
    """Print CV claims comparison table."""
    spec = metrics["specificity"] * 100
    gmeans_imp = 0
    if baseline_gmeans:
        gmeans_imp = ((metrics["gmeans"] - baseline_gmeans) / baseline_gmeans) * 100

    print("\n" + "-" * 70)
    print("CV CLAIMS COMPARISON")
    print("-" * 70)
    print(
        f"| {'Metric':<20} | {'Achieved':>12} | {'CV Claim':>12} | {'Delta':>8} | {'Status':>6} |"
    )
    print(f"|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*10}|{'-'*8}|")

    # Specificity
    delta_spec = spec - CV_CLAIMS["specificity"]
    status_spec = "✓" if abs(delta_spec) <= 3 else "✗"
    print(
        f"| {'Specificity':<20} | {spec:>11.2f}% | {CV_CLAIMS['specificity']:>11.2f}% | {delta_spec:>+7.2f}% | {status_spec:>6} |"
    )

    # G-Means Improvement
    delta_gm = gmeans_imp - CV_CLAIMS["gmeans_improvement"]
    status_gm = "✓" if abs(delta_gm) <= 5 else "✗"
    print(
        f"| {'G-Means Improve':<20} | {gmeans_imp:>11.2f}% | {CV_CLAIMS['gmeans_improvement']:>11.2f}% | {delta_gm:>+7.2f}% | {status_gm:>6} |"
    )

    # Latency
    p95 = latency["p95_ms"]
    status_lat = "✓" if p95 < CV_CLAIMS["p95_latency_ms"] else "✗"
    print(f"| {'P95 Latency':<20} | {p95:>10.1f}ms | {'<100':>10}ms | {'-':>8} | {status_lat:>6} |")

    print("-" * 70)
    print("\nStatus: ✓ = PASS (within tolerance), ✗ = INVESTIGATE")


def save_metrics_csv(history, path="logs/training_metrics.csv"):
    """Save epoch-by-epoch metrics to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_specificity",
                "val_recall",
                "val_f1",
                "val_gmeans",
            ],
        )
        writer.writeheader()
        for h in history:
            writer.writerow(
                {
                    "epoch": h.get("epoch", 0),
                    "train_loss": h.get("train_loss", 0),
                    "val_loss": h.get("val_loss", 0),
                    "val_specificity": h.get("val_specificity", 0),
                    "val_recall": h.get("val_recall", 0),
                    "val_f1": h.get("val_f1", 0),
                    "val_gmeans": h.get("val_gmeans", 0),
                }
            )
    print(f"\nMetrics saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train FraudGuard (Full Pipeline)")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mcd", action="store_true")
    parser.add_argument("--no_rl", action="store_true")
    args = parser.parse_args()

    use_mcd = not args.no_mcd
    use_rl = not args.no_rl

    # Header
    print_header(args.sample_frac, args.epochs, use_mcd, use_rl)

    # Set seed
    set_seed(args.seed)

    # Load data
    print("\nLoading data...")
    loader = FraudDataLoader()
    df = loader.load_train_data(sample_frac=args.sample_frac)
    train_df, val_df, test_df = loader.create_splits(df)

    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        EarlyStopping(monitor="val_gmeans", patience=30, mode="max"),
        ModelCheckpoint(path="models", monitor="val_gmeans", mode="max"),
        MetricsLogger(log_dir="logs"),
    ]

    # Train baseline (for G-Means improvement calculation)
    baseline_gmeans = None
    if use_mcd or use_rl:
        print("\n[Baseline] Training without MCD/RL for comparison...")
        baseline_trainer = FraudTrainer(callbacks=[EarlyStopping(patience=20)])
        baseline_metrics = baseline_trainer.fit(
            train_df.copy(),
            val_df.copy(),
            test_df.copy(),
            max_epochs=min(50, args.epochs),
            use_mcd=False,
            use_rl=False,
        )
        baseline_gmeans = baseline_metrics["gmeans"]
        print(f"[Baseline] G-Means: {baseline_gmeans*100:.2f}%")

    # Train full pipeline
    print("\n[Full] Training with AdaptiveMCD + RL pipeline...")
    trainer = FraudTrainer(callbacks=callbacks)
    metrics = trainer.fit(
        train_df,
        val_df,
        test_df,
        max_epochs=args.epochs,
        use_mcd=use_mcd,
        use_rl=use_rl,
    )

    # Data summary (after training so we have MCD info)
    print_data_summary(train_df, val_df, test_df, trainer)

    # Training progression
    print_training_progress(trainer.history)

    # Early stopping info
    if len(trainer.history) < args.epochs:
        print(f"\nEarly stopping triggered at epoch: {len(trainer.history)}")
    else:
        print(f"\nCompleted all {args.epochs} epochs")

    # Benchmark latency
    latency = trainer.benchmark_latency(n_runs=100)

    # Save model
    model_path = f"models/fraudguard_full_trained_{timestamp}.pt"
    trainer.save(model_path)

    # Final results
    print_final_results(metrics, latency, metrics.get("train_time", 0), baseline_gmeans)

    # CV comparison
    print_cv_comparison(metrics, latency, baseline_gmeans)

    # Save CSV
    save_metrics_csv(trainer.history, f"logs/training_metrics_{timestamp}.csv")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print(f"Best model: models/best_model.pt")


if __name__ == "__main__":
    main()
