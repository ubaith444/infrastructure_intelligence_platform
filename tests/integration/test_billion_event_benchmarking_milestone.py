"""Milestone 4 Acceptance Test Suite: Billion-Event Benchmarking.

Verification criteria:
1. High-throughput multithreaded stress testing and pipeline lag measurements.
2. Vectorized DuckDB Lakehouse analytics scan performance and 1-billion event scaling projections.
3. Empirical single-node capacity limits vs. distributed clustering migration decision matrix.
4. API endpoints: /api/v1/benchmarks/report, /api/v1/benchmarks/stress-test,
   /api/v1/benchmarks/billion-event-projections, /api/v1/benchmarks/decision-matrix.
"""

from fastapi.testclient import TestClient
import pytest

from apps.api.server import app
from data_science.evaluation.benchmark_evaluator import BenchmarkEvaluator


@pytest.fixture
def client():
    return TestClient(app)


def test_multithreaded_stress_test_throughput_and_lag():
    """Verify concurrent multithreaded producers and consumer lag drain dynamics."""
    evaluator = BenchmarkEvaluator()
    result = evaluator.run_multithreaded_stress_test(
        total_events=2000,
        num_producers=4,
        num_consumers=2,
        batch_size=50,
    )

    assert result["events_benchmarked"] >= 1800
    assert result["pipeline_lag"]["events_consumed"] == result["events_benchmarked"]
    assert result["producer_threads"] == 4
    assert result["consumer_threads"] == 2
    assert result["duration_seconds"] > 0
    assert result["throughput_eps"] > 0
    assert result["batch_latency_p50_ms"] >= 0
    assert result["batch_latency_p99_ms"] >= result["batch_latency_p50_ms"]
    assert result["single_node_safe_capacity_eps"] >= 50000.0
    assert result["architecture_verdict"] == "SINGLE_NODE_CAPACITY_SUFFICIENT"

    # Pipeline lag validation
    lag = result["pipeline_lag"]
    assert lag["peak_consumer_lag"] >= 0
    assert lag["events_consumed"] >= 0
    assert lag["lag_drain_rate_eps"] > 0
    assert lag["catch_up_status"] in ("DRAINED", "DRAINING")


def test_duckdb_vectorized_billion_event_scaling_projections():
    """Verify DuckDB analytical query execution and 1-billion event footprint extrapolations."""
    evaluator = BenchmarkEvaluator()
    result = evaluator.benchmark_duckdb_billion_event_analytics(sample_rows=2000)

    assert result["sample_rows_tested"] == 2000
    assert result["query_latency_ms"] > 0
    assert result["scan_rate_rows_per_second"] > 0
    assert result["aggregation_groups_returned"] > 0
    assert result["status"] == "PASS"

    projections = result["billion_event_projections"]
    # 1 Billion uncompressed ~160-185 GB
    assert 140.0 <= projections["uncompressed_raw_volume_gb"] <= 200.0
    # 1 Billion Snappy Parquet ~40-60 GB (fits on single commodity NVMe)
    assert 35.0 <= projections["compressed_columnar_parquet_gb"] <= 65.0
    assert projections["projected_single_node_ingest_hours"] > 0
    assert projections["projected_distributed_cluster_ingest_hours"] > 0
    assert "4.8x" in projections["distributed_tco_cost_multiplier"]


def test_distributed_decision_matrix_boundary_thresholds():
    """Verify mathematical & operational boundary conditions between single-node and distributed clusters."""
    evaluator = BenchmarkEvaluator()

    # Case A: Typical single-node workload (15,000 EPS, p99=2.5ms)
    matrix_single = evaluator.generate_distributed_decision_matrix(
        measured_eps=15000.0,
        measured_latency_p99_ms=2.5,
    )
    assert matrix_single["recommendation"] == "RETAIN_SINGLE_NODE_ARCHITECTURE"
    assert matrix_single["headroom_factor"] >= 2.0
    assert "RETAIN" in matrix_single["recommendation"]

    triggers = matrix_single["distributed_migration_triggers"]
    assert len(triggers) == 4
    for trig in triggers:
        assert trig["verdict"] == "SINGLE_NODE_OPTIMAL"

    # Case B: Extreme scale requiring distributed clustering (110,000 EPS, p99=45.0ms)
    matrix_dist = evaluator.generate_distributed_decision_matrix(
        measured_eps=110000.0,
        measured_latency_p99_ms=45.0,
    )
    assert matrix_dist["recommendation"] == "MIGRATE_TO_DISTRIBUTED_CLUSTER"


def test_v1_benchmark_report_api(client):
    """Verify GET /api/v1/benchmarks/report endpoint."""
    res = client.get("/api/v1/benchmarks/report")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark_id" in data
    assert "ingestion_performance" in data
    assert "multithreaded_stress_test" in data
    assert "billion_event_duckdb_analytics" in data
    assert "anomaly_detection_smd_benchmark" in data
    assert "distributed_processing_decision_matrix" in data
    assert data["overall_status"] == "PASS"


def test_v1_multithreaded_stress_test_api(client):
    """Verify POST /api/v1/benchmarks/stress-test endpoint."""
    res = client.post("/api/v1/benchmarks/stress-test?total_events=1000&num_producers=2&num_consumers=1")
    assert res.status_code == 200
    data = res.json()
    assert data["events_benchmarked"] >= 900
    assert data["producer_threads"] == 2
    assert data["throughput_eps"] > 0
    assert "pipeline_lag" in data
    assert data["pipeline_lag"]["events_consumed"] == data["events_benchmarked"]


def test_v1_billion_event_projections_api(client):
    """Verify GET /api/v1/benchmarks/billion-event-projections endpoint."""
    res = client.get("/api/v1/benchmarks/billion-event-projections")
    assert res.status_code == 200
    data = res.json()
    assert "billion_event_projections" in data
    assert data["status"] == "PASS"
    assert data["billion_event_projections"]["compressed_columnar_parquet_gb"] > 0


def test_v1_decision_matrix_api(client):
    """Verify GET /api/v1/benchmarks/decision-matrix endpoint."""
    res = client.get("/api/v1/benchmarks/decision-matrix?measured_eps=20000&measured_p99_ms=3.0")
    assert res.status_code == 200
    data = res.json()
    assert data["recommendation"] == "RETAIN_SINGLE_NODE_ARCHITECTURE"
    assert "distributed_migration_triggers" in data
    assert len(data["distributed_migration_triggers"]) == 4
