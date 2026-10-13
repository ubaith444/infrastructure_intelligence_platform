import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from storage.models import MetricWindow


class MultivariateIsolationForest:
    """Multivariate unsupervised anomaly detector with subspace feature attribution.
    
    Evaluates 5 operational dimensions:
    - latency_p90
    - error_rate
    - request_rate
    - cpu_utilization
    - memory_utilization
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 60):
        self._lock = threading.Lock()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=1,
        )

        self.fitted = False
        self.feature_names = ["latency_p90", "error_rate", "request_rate", "cpu_utilization", "memory_utilization"]
        self.baseline_stats: Dict[str, Dict[str, float]] = {
            "latency_p90": {"median": 45.0, "mad": 15.0},
            "error_rate": {"median": 0.005, "mad": 0.003},
            "request_rate": {"median": 120.0, "mad": 25.0},
            "cpu_utilization": {"median": 35.0, "mad": 10.0},
            "memory_utilization": {"median": 45.0, "mad": 8.0},
        }

    def fit_baseline(self, baseline_samples: np.ndarray) -> None:
        """Fit Isolation Forest on historical nominal observation matrix."""
        with self._lock:
            if len(baseline_samples) >= 20:
                self.model.fit(baseline_samples)
                self.fitted = True
                # Update baseline stats for attribution
                for idx, feat in enumerate(self.feature_names):
                    col = baseline_samples[:, idx]
                    med = float(np.median(col))
                    mad = float(np.median(np.abs(col - med)))
                    self.baseline_stats[feat] = {"median": med, "mad": max(1e-4, mad)}

    def score_window(self, window: MetricWindow) -> Tuple[float, bool, Dict[str, float]]:
        vec = np.array([[
            window.latency_p90,
            window.error_rate * 100.0,
            window.request_rate,
            window.cpu_utilization,
            window.memory_utilization,
        ]])

        with self._lock:
            if not self.fitted:
                # Synthetic fit on nominal operating values
                np.random.seed(42)
                synth = np.array([
                    np.random.normal(45.0, 10.0, 60),      # latency
                    np.random.normal(0.5, 0.2, 60),        # error rate * 100
                    np.random.normal(120.0, 20.0, 60),     # request rate
                    np.random.normal(35.0, 8.0, 60),       # cpu
                    np.random.normal(45.0, 6.0, 60),       # memory
                ]).T
                self.model.fit(synth)
                self.fitted = True

            raw_score = -float(self.model.score_samples(vec)[0])  # higher = more anomalous

        norm_score = max(0.0, min(1.0, (raw_score - 0.40) / 0.40))
        is_anom = norm_score > 0.65

        # Subspace feature attribution: quantify how much each dimension deviates from nominal baseline
        deviations = {}
        for feat in self.feature_names:
            val = getattr(window, feat, 0.0)
            if feat == "error_rate":
                val = val * 100.0
            stat = self.baseline_stats.get(feat, {"median": 10.0, "mad": 5.0})
            med = stat["median"]
            mad = stat["mad"]
            dev = max(0.0, abs(val - med) / (1.4826 * mad))
            deviations[feat] = dev

        total_dev = sum(deviations.values()) + 1e-6
        attributions = {
            feat: round(deviations[feat] / total_dev, 3)
            for feat in self.feature_names
        }

        return float(round(norm_score, 3)), is_anom, attributions

