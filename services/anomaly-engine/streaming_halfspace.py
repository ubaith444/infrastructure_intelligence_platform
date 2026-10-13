from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from river import anomaly, drift
from storage.models import MetricWindow


class RiverStreamingDetector:
    """Online incremental HalfSpaceTrees detector and ADWIN change detector with incident freeze safeguard."""

    def __init__(self, n_trees: int = 25, height: int = 8, window_size: int = 150):
        self.model = anomaly.HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=42,
        )
        self.threshold = 0.70
        self.total_learned = 0
        self.freeze_count = 0
        # ADWIN Change Detectors
        self.adwin_lat = drift.ADWIN()
        self.adwin_err = drift.ADWIN()
        self.drift_detected = False

    def score_and_learn(
        self,
        window: MetricWindow,
        is_incident_active: bool = False,
    ) -> Tuple[float, bool, bool, Dict[str, float]]:
        features = {
            "lat": window.latency_p90,
            "err": window.error_rate * 100.0,
            "req": window.request_rate,
            "cpu": window.cpu_utilization,
        }

        score = float(self.model.score_one(features))
        is_anom = score >= self.threshold

        # Update ADWIN change detectors
        self.adwin_lat.update(window.latency_p90)
        self.adwin_err.update(window.error_rate * 100.0)
        drift_lat = self.adwin_lat.drift_detected
        drift_err = self.adwin_err.drift_detected
        self.drift_detected = drift_lat or drift_err

        # Incident Freeze Safeguard: Do NOT learn anomalous patterns during active failure
        if not is_incident_active and not is_anom:
            self.model.learn_one(features)
            self.total_learned += 1
        else:
            self.freeze_count += 1

        attributions = {
            "latency": 0.5 if drift_lat else (0.35 if score > 0.5 else 0.25),
            "error_rate": 0.5 if drift_err else (0.35 if score > 0.5 else 0.25),
            "request_rate": 0.15,
            "cpu_utilization": 0.15,
        }
        tot = sum(attributions.values())
        norm_attributions = {k: round(v / tot, 3) for k, v in attributions.items()}

        return float(round(score, 4)), is_anom, self.drift_detected, norm_attributions

