import numpy as np
import pytest
from services.anomaly_engine.statistical_mad import RobustStatisticalDetector

def test_statistical_mad_nominal_and_outlier():
    det = RobustStatisticalDetector(threshold_z=3.5)
    # Feed baseline
    for val in np.random.normal(50.0, 2.0, 25):
        det.score("checkout-api", "latency", float(val))

    # Test nominal point
    z_nom, med, is_anom = det.score("checkout-api", "latency", 51.0)
    assert is_anom is False
    assert z_nom < 3.5

    # Test extreme outlier
    z_out, _, is_anom_out = det.score("checkout-api", "latency", 250.0)
    assert is_anom_out is True
    assert z_out > 3.5


def test_zero_mad_safe_fallback():
    det = RobustStatisticalDetector()
    # Feed constant values where MAD would be 0
    for _ in range(20):
        det.score("auth-service", "zero_metric", 10.0)

    # Should not throw ZeroDivisionError
    z_score, _, is_anom = det.score("auth-service", "zero_metric", 10.0)
    assert z_score == 0.0
    assert is_anom is False
