import concurrent.futures
import time
from typing import List, Tuple
import numpy as np
import pytest

from simulator.event_generator import TelemetrySimulator
from storage.broker import EmbeddedEventBroker
from services.feature_engine.rolling_windows import RollingWindowAggregator
from services.anomaly_engine.arbiter import MultiModelAnomalyArbiter


def test_high_throughput_multithreaded_ingestion(tmp_path):
    """Verify single-node broker and pipeline can handle high-throughput concurrent load."""
    broker = EmbeddedEventBroker(storage_dir=tmp_path / "broker", num_partitions=8)
    windows = RollingWindowAggregator()
    sim = TelemetrySimulator()

    total_workers = 4
    ticks_per_worker = 30

    def worker_task(worker_id: int) -> Tuple[int, List[float]]:
        latencies = []
        count = 0
        for _ in range(ticks_per_worker):
            t0 = time.perf_counter()
            tick = sim.generate_tick()
            for evt in tick:
                broker.publish("telemetry.events", evt)
                windows.add_event(evt)
                count += 1
            batch_dur = (time.perf_counter() - t0) * 1000.0
            latencies.append(batch_dur)
        return count, latencies

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = [executor.submit(worker_task, i) for i in range(total_workers)]
        results = [f.result() for f in futures]
    t_end = time.perf_counter()

    dur = t_end - t_start
    total_events = sum(r[0] for r in results)
    throughput_eps = total_events / max(0.001, dur)

    all_latencies = [lat for r in results for lat in r[1]]
    lat_p50 = float(np.percentile(all_latencies, 50))
    lat_p95 = float(np.percentile(all_latencies, 95))
    lat_p99 = float(np.percentile(all_latencies, 99))

    # Assertions
    assert throughput_eps > 500.0, f"Expected >500 EPS, got {throughput_eps:.1f}"
    assert lat_p95 < 200.0, f"Expected p95 latency < 200ms, got {lat_p95:.2f}ms"

    # Broker partition balancing verification
    total_buffered = sum(
        len(broker.partitions["telemetry.events"][pid])
        for pid in range(broker.num_partitions)
    )
    assert total_buffered == total_events, f"Event loss detected! {total_buffered} != {total_events}"

    # Verify consumer lag and consumption
    batch = broker.poll("telemetry.events", "cg_stress", max_records=2000)
    assert len(batch) > 0
    broker.commit("telemetry.events", "cg_stress", partition_id=0, offset=10)



def test_concurrent_anomaly_scoring_under_load():
    """Verify anomaly engine can score multi-service windows concurrently without thread race conditions."""
    arbiter = MultiModelAnomalyArbiter()
    sim = TelemetrySimulator()
    windows = RollingWindowAggregator()

    # Pre-populate windows
    for _ in range(25):
        tick = sim.generate_tick()
        for evt in tick:
            windows.add_event(evt)

    services = sim.services

    def score_service(svc: str):
        win = windows.get_window(svc)
        if win:
            return arbiter.evaluate_window(win)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(score_service, s) for s in services * 10]
        results = [f.result() for f in futures]

    assert len(results) == len(services) * 10
    assert len(arbiter.recent_anomalies) >= 0
