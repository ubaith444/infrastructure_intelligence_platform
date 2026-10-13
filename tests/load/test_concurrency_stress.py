import time
import pytest
from simulator.event_generator import TelemetrySimulator
from storage.broker import EmbeddedEventBroker

def test_load_concurrency_stress():
    broker = EmbeddedEventBroker(num_partitions=4)
    sim = TelemetrySimulator()

    t0 = time.perf_counter()
    count = 500
    for _ in range(count // 20):
        tick = sim.generate_tick()
        for evt in tick:
            broker.publish("telemetry.events", evt)
    dur = time.perf_counter() - t0

    throughput_eps = count / max(0.001, dur)
    assert throughput_eps > 100.0
    lag = broker.get_lag("telemetry.events", "cg_load")
    assert lag["total_lag"] > 0
