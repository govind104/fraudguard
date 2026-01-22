"""Metrics computation and evaluation for fraud detection.

Provides metrics: Specificity, Recall, F1, G-Means, and latency benchmarking.

Example:
    >>> evaluator = Evaluator()
    >>> metrics = evaluator.compute_metrics(y_true, y_pred)
    >>> evaluator.print_report(metrics)
"""

import time
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Fraud detection metrics evaluator.

    Computes:
    - Specificity (True Negative Rate)
    - Recall (True Positive Rate / Sensitivity)
    - F1 Score
    - G-Means (geometric mean of TPR and TNR)
    - Precision
    """

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of metric name to value.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity

        # G-Means
        gmeans = np.sqrt(tpr * tnr) if (tpr > 0 and tnr > 0) else 0.0

        # F1 and Precision
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        return {
            "specificity": tnr,
            "recall": tpr,
            "f1": f1,
            "gmeans": gmeans,
            "precision": precision,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    def compute_gmeans_improvement(
        self,
        baseline_gmeans: float,
        model_gmeans: float,
    ) -> float:
        """Compute G-Means improvement percentage.

        Args:
            baseline_gmeans: G-Means of baseline model.
            model_gmeans: G-Means of improved model.

        Returns:
            Improvement percentage.
        """
        if baseline_gmeans == 0:
            return 0.0
        return ((model_gmeans - baseline_gmeans) / baseline_gmeans) * 100

    def benchmark_latency(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference latency.

        Args:
            model: Trained model.
            x: Node features.
            edge_index: Graph edges.
            n_runs: Number of inference runs.

        Returns:
            Latency statistics in milliseconds.
        """
        model.eval()
        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, edge_index)

        # Benchmark
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(x, edge_index)
                latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }

    def print_report(
        self,
        metrics: Dict[str, float],
        title: str = "Evaluation Results",
    ) -> None:
        """Print formatted metrics report."""
        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        print(f"| {'Metric':<20} | {'Value':>15} |")
        print(f"|{'-'*22}|{'-'*17}|")

        for key in ["specificity", "recall", "f1", "gmeans", "precision"]:
            if key in metrics:
                print(f"| {key.capitalize():<20} | {metrics[key]*100:>14.2f}% |")

        print("=" * 50)
        print(
            f"Confusion: TP={metrics.get('tp', 0)}, TN={metrics.get('tn', 0)}, "
            f"FP={metrics.get('fp', 0)}, FN={metrics.get('fn', 0)}"
        )

    def print_cv_comparison(
        self,
        achieved: Dict[str, float],
        cv_claims: Dict[str, float],
    ) -> None:
        """Print comparison against CV claims."""
        print("\n" + "=" * 60)
        print("CV METRICS REPRODUCTION")
        print("=" * 60)
        print(f"| {'Metric':<20} | {'Achieved':>12} | {'CV Claim':>12} | {'Match':>6} |")
        print(f"|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*8}|")

        for key, target in cv_claims.items():
            value = achieved.get(key, 0)

            # Format based on metric type
            if key in ["specificity", "gmeans_improvement"]:
                val_str = f"{value:.2f}%"
                target_str = f"{target:.2f}%"
                match = "✓" if value >= target * 0.95 else "✗"  # 5% tolerance
            elif key == "p95_latency_ms":
                val_str = f"{value:.0f}ms"
                target_str = f"<{target}ms"
                match = "✓" if value <= target else "✗"
            else:
                val_str = f"{value:.4f}"
                target_str = f"{target:.4f}"
                match = "✓" if abs(value - target) < 0.05 else "✗"

            print(f"| {key:<20} | {val_str:>12} | {target_str:>12} | {match:>6} |")

        print("=" * 60)
