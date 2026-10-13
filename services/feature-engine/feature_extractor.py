from typing import Any, Dict
from storage.models import MetricWindow


class FeatureExtractor:
    """Extracts rate-of-change, velocity, and pressure signals for prediction and anomaly scoring."""

    @staticmethod
    def extract_features(window: MetricWindow) -> Dict[str, float]:
        return {
            "req_volume": window.request_rate,
            "error_rate": window.error_rate,
            "latency_p50": window.latency_p50,
            "latency_p90": window.latency_p90,
            "latency_p99": window.latency_p99,
            "latency_spread": max(0.0, window.latency_p90 - window.latency_p50),
            "cpu_pressure": window.cpu_utilization,
            "mem_pressure": window.memory_utilization,
            "saturation_ratio": window.request_rate / 350.0,
        }
