import numpy as np
import pytest
from data_science.drift.psi_monitor import ModelHealthMonitor

def test_psi_monitor_drift():
    mon = ModelHealthMonitor()

    # Identical distributions -> low PSI
    d1 = np.random.normal(100, 10, 200)
    d2 = np.random.normal(100, 10, 200)
    psi_stable = mon.compute_psi(d1, d2)
    assert psi_stable < 0.10

    # Shifted distributions -> high PSI
    d3 = np.random.normal(180, 10, 200)
    psi_drifted = mon.compute_psi(d1, d3)
    assert psi_drifted >= 0.20
