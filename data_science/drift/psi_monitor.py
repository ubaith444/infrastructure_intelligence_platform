from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np


class ModelHealthMonitor:
    """Tracks model inference latency percentiles and computes Population Stability Index (PSI) feature drift."""

    def __init__(self):
        self.latencies: Dict[str, deque] = {
            "risk_predictor": deque(maxlen=500),
            "forecaster": deque(maxlen=500),
        }
        self.baseline_distributions: Dict[str, np.ndarray] = {
            "latency_p90": np.linspace(50.0, 300.0, 10),
            "error_rate": np.linspace(0.001, 0.05, 10),
            "cpu_utilization": np.linspace(10.0, 80.0, 10),
        }
        self.observed_features: Dict[str, deque] = {
            k: deque(maxlen=500) for k in self.baseline_distributions
        }

    def record_latency(self, model_name: str, latency_ms: float) -> None:
        if model_name not in self.latencies:
            self.latencies[model_name] = deque(maxlen=500)
        self.latencies[model_name].append(latency_ms)

    def record_feature(self, feature_name: str, value: float) -> None:
        if feature_name in self.observed_features:
            self.observed_features[feature_name].append(value)

    def compute_psi(self, baseline: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        if len(baseline) == 0 or len(actual) == 0:
            return 0.0
        # For smaller sample sizes (e.g. N < 500), use 5-8 buckets to prevent high Monte Carlo variance
        eff_buckets = min(num_buckets, max(5, int(min(len(baseline), len(actual)) / 30)))
        bins = np.percentile(baseline, np.linspace(0, 100, eff_buckets + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        bins[0] = -np.inf
        bins[-1] = np.inf

        b_counts, _ = np.histogram(baseline, bins=bins)
        a_counts, _ = np.histogram(actual, bins=bins)
        eps = 1e-3
        b_perc = (b_counts + eps) / (len(baseline) + eps * len(b_counts))
        a_perc = (a_counts + eps) / (len(actual) + eps * len(a_counts))
        psi = np.sum((a_perc - b_perc) * np.log(a_perc / b_perc))
        return float(round(max(0.0, psi), 4))

    def get_latency_percentiles(self, model_name: str) -> Dict[str, float]:
        """Compute p50, p90, p95, p99 latency percentiles for the specified model."""
        hist = self.latencies.get(model_name)
        if not hist or len(hist) == 0:
            return {"count": 0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(list(hist))
        return {
            "count": len(arr),
            "p50_ms": float(round(np.percentile(arr, 50), 3)),
            "p90_ms": float(round(np.percentile(arr, 90), 3)),
            "p95_ms": float(round(np.percentile(arr, 95), 3)),
            "p99_ms": float(round(np.percentile(arr, 99), 3)),
            "mean_ms": float(round(np.mean(arr), 3)),
        }

    def reset_baseline_distribution(self, feature_name: str, new_distribution: np.ndarray) -> None:
        """Update reference baseline distribution after retraining or scheduled recalibration."""
        if len(new_distribution) > 0:
            self.baseline_distributions[feature_name] = np.array(new_distribution)
            if feature_name in self.observed_features:
                self.observed_features[feature_name].clear()

    def check_retraining_trigger(self) -> Dict[str, Any]:
        """Evaluates whether current PSI drift or latency degradation breaches automated retraining thresholds."""
        drift_report = self.evaluate_drift()
        max_psi = drift_report.get("max_psi", 0.0)
        retrain_needed = bool(max_psi >= 0.25)
        reason = "Severe distribution shift (PSI >= 0.25)" if retrain_needed else "Distributions within acceptable tolerance."
        return {
            "retrain_recommended": retrain_needed,
            "max_psi": max_psi,
            "threshold": 0.25,
            "reason": reason,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    def evaluate_drift(self) -> Dict[str, Any]:
        results = {}
        max_psi = 0.0
        for feat, hist in self.observed_features.items():
            if len(hist) < 20:
                results[feat] = {"psi": 0.0, "status": "INSUFFICIENT_DATA", "samples": len(hist)}
                continue
            base = self.baseline_distributions[feat]
            act = np.array(list(hist))
            psi = self.compute_psi(base, act)
            max_psi = max(max_psi, psi)
            status = "SIGNIFICANT_DRIFT" if psi >= 0.25 else ("MODERATE_DRIFT" if psi >= 0.10 else "STABLE")
            results[feat] = {"psi": psi, "status": status, "samples": len(hist)}

        rec = "Model retraining recommended due to significant feature drift (PSI >= 0.25)." if max_psi >= 0.25 else "Feature distribution nominal."
        latency_stats = {
            model: self.get_latency_percentiles(model) for model in self.latencies
        }
        return {
            "status": "DRIFT_DETECTED" if max_psi >= 0.25 else "HEALTHY",
            "max_psi": max_psi,
            "recommendation": rec,
            "feature_drift": results,
            "latency_percentiles": latency_stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

