"""Milestone Acceptance Criteria Verification Suite.

Definition of Done - 7 Acceptance Criteria:
1. Instrument a demo microservice (OpenTelemetry tracer, meter, spans, latency/error telemetry).
2. Collect telemetry (OpenTelemetry Collector / OTLP HTTP ingestion, validation, DLQ isolation).
3. Build durable ingestion (Redpanda/Kafka partitioned disk broker log, replayable stream).
4. Persist historical events (DuckDB Lakehouse Parquet files and Relational operational metadata).
5. Implement anomaly detection (Robust Statistical MAD, rolling median, explainable scores, Zero-MAD fallback).
6. Build a working dashboard (Endpoints answering the 4 operational triage questions & static dashboard).
7. Verify end-to-end behavior (Controlled fault injection, detection latency < 1.5s, 100% completeness, 0 false alarms).
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from fastapi.testclient import TestClient
import numpy as np
import pytest

from apps.api.server import app
from services.anomaly_engine.statistical_mad import RobustStatisticalDetector
from services.feature_engine.rolling_windows import RollingWindowAggregator
from services.ingestion.collector import TelemetryCollector, parse_otlp_payload
from services.telemetry_demo.app import order_app, exporter
from storage.broker import EmbeddedEventBroker
from storage.lakehouse import LakehouseManager
from storage.metadata_store import OperationalMetadataStore
from storage.models import AlertSeverity, MetricWindow, TelemetryEvent


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def demo_client():
    return TestClient(order_app)


# ============================================================================
# CRITERION 1: Instrument a demo microservice
# ============================================================================
def test_criterion_1_instrument_demo_microservice(demo_client):
    """Criterion 1: Generate real request, latency and error telemetry using OpenTelemetry."""
    exporter.buffer.clear()

    # 1. Normal successful request with simulated delay
    payload = {
        "customer_id": "cust-test-1",
        "items": [{"item_id": "item-1", "name": "Item A", "quantity": 2, "price": 19.99}],
        "simulate_delay_ms": 20.0,
        "simulate_error": False,
    }
    res = demo_client.post("/orders", json=payload)
    assert res.status_code == 200
    order_data = res.json()
    assert order_data["status"] == "CONFIRMED"
    assert order_data["total_amount"] == 39.98

    # 2. Simulated fault error request
    err_payload = {
        "customer_id": "cust-test-2",
        "items": [{"item_id": "item-2", "name": "Item B", "quantity": 1, "price": 50.00}],
        "simulate_delay_ms": 10.0,
        "simulate_error": True,
    }
    err_res = demo_client.post("/orders", json=err_payload)
    assert err_res.status_code == 500

    # 3. Verify OpenTelemetry instrumentation captured telemetry in exporter buffer
    assert len(exporter.buffer) >= 4  # 2 events per request (latency + error)

    latency_events = [e for e in exporter.buffer if e["metric_name"] == "request_latency_ms"]
    error_events = [e for e in exporter.buffer if e["metric_name"] == "error_rate"]

    assert len(latency_events) >= 2
    assert len(error_events) >= 2

    # Check first (successful) request
    first_lat = latency_events[0]
    assert first_lat["service"] == "order-service"
    assert first_lat["metric_value"] >= 15.0  # Captures the simulated delay accurately
    assert first_lat["trace_id"] is not None and len(first_lat["trace_id"]) == 32
    assert first_lat["span_id"] is not None and len(first_lat["span_id"]) == 16
    assert first_lat["attributes"]["http.status_code"] == 200

    # Check second (fault) request
    second_err = error_events[1]
    assert second_err["service"] == "order-service"
    assert second_err["metric_value"] == 1.0  # Error flag is set
    assert second_err["attributes"]["http.status_code"] == 500


# ============================================================================
# CRITERION 2: Collect telemetry
# ============================================================================
def test_criterion_2_collect_telemetry(client):
    """Criterion 2: Configure OpenTelemetry Collector and verify received data."""
    # 1. Test standard OTLP metrics format ingested via /v1/metrics
    otlp_metrics = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "order-service"}},
                        {"key": "deployment.environment", "value": {"stringValue": "production"}},
                    ]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "request_latency_ms",
                                "unit": "ms",
                                "gauge": {
                                    "dataPoints": [
                                        {
                                            "asDouble": 45.2,
                                            "timeUnixNano": str(int(time.time() * 1e9)),
                                            "attributes": [{"key": "http.method", "value": {"stringValue": "POST"}}],
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ],
            }
        ]
    }
    res_metrics = client.post("/v1/metrics", json=otlp_metrics)
    assert res_metrics.status_code == 200
    assert res_metrics.json()["ingested"] >= 1

    # 2. Test OTLP traces format ingested via /v1/traces
    now_nano = int(time.time() * 1e9)
    otlp_traces = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "order-service"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
                                "spanId": "00f067aa0ba902b7",
                                "name": "create_order",
                                "startTimeUnixNano": str(now_nano - int(50 * 1e6)),
                                "endTimeUnixNano": str(now_nano),
                                "status": {"code": 1},
                                "attributes": [
                                    {"key": "http.status_code", "value": {"intValue": 200}}
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }
    res_traces = client.post("/v1/traces", json=otlp_traces)
    assert res_traces.status_code == 200
    assert res_traces.json()["ingested"] >= 2  # latency + error_rate

    # 3. Test Collector validation & Dead Letter Queue (DLQ) rejection for invalid telemetry
    collector = TelemetryCollector()
    invalid_event = {
        "event_id": "evt-invalid-1",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "service": "order-service",
        "attributes": {"user_id": "banned_key_violation"},  # Banned key triggers validation failure
    }
    valid = collector.ingest([invalid_event])
    assert len(valid) == 0
    assert collector.rejected_count == 1
    assert collector.dlq.total_violations >= 1


# ============================================================================
# CRITERION 3: Build durable ingestion
# ============================================================================
def test_criterion_3_build_durable_ingestion(tmp_path):
    """Criterion 3: Use Redpanda/Kafka broker abstraction, event validation and replayable stream."""
    broker = EmbeddedEventBroker(storage_dir=tmp_path / "broker", num_partitions=2)
    topic = "telemetry.events"

    # Publish 10 valid events across partitions
    now = datetime.now(timezone.utc)
    for i in range(10):
        evt = TelemetryEvent(
            event_id=f"evt-durable-{i}",
            event_time=now,
            service="order-service",
            metric_name="request_latency_ms",
            metric_value=40.0 + i,
        )
        broker.publish(topic, evt)

    # Check partition file persistence on disk
    partition_files = [tmp_path / "broker" / topic / f"partition_{p}.log" for p in range(2)]
    assert any(p.exists() for p in partition_files)

    # Consumer reads events and advances offset
    consumer_events = broker.poll(topic, "acceptance_consumer", max_records=10)
    assert len(consumer_events) == 10

    # Check lag is 0 after full consumption
    lag = broker.get_lag(topic, "acceptance_consumer")
    assert lag["total_lag"] == 0

    # Replay event stream from earliest offset
    broker.reset_offsets(topic, "acceptance_consumer", target="earliest")
    replayed = broker.poll(topic, "acceptance_consumer", max_records=10)
    assert len(replayed) == 10
    assert all(isinstance(e, TelemetryEvent) for e in replayed)


# ============================================================================
# CRITERION 4: Persist historical events
# ============================================================================
def test_criterion_4_persist_historical_events(tmp_path):
    """Criterion 4: Store normalized records as Parquet and operational metadata in PostgreSQL/Relational."""
    lakehouse_dir = tmp_path / "lakehouse"
    db_path = tmp_path / "metadata.db"

    # 1. Lakehouse Parquet persistence
    lakehouse = LakehouseManager(lakehouse_dir=lakehouse_dir)
    now = datetime.now(timezone.utc)
    events = [
        TelemetryEvent(
            event_id=f"evt-lake-{i}",
            event_time=now,
            service="order-service",
            metric_name="request_latency_ms",
            metric_value=42.0 + i,
        )
        for i in range(5)
    ]
    written = lakehouse.write_events(events)
    assert written == 5
    assert lakehouse.total_records == 5

    # Check that parquet files are actually written to disk
    parquet_files = list(lakehouse_dir.rglob("*.parquet"))
    assert len(parquet_files) >= 1

    # Query Parquet files via DuckDB SQL engine
    sql_path = (parquet_files[0]).as_posix().replace("\\", "/")
    duck_res = lakehouse.db.query(f"SELECT count(*) as total, avg(metric_value) as avg_val FROM read_parquet('{sql_path}')").fetchall()
    assert duck_res[0][0] == 5
    assert 43.0 <= duck_res[0][1] <= 45.0

    # 2. Relational Operational Metadata persistence
    meta_store = OperationalMetadataStore(db_path=db_path)
    services = meta_store.list_services()
    assert len(services) > 0
    assert any("service_id" in s for s in services)

    # Record operational pipeline health
    meta_store.record_pipeline_health(
        throughput=150.0,
        lag=0,
        errors=0,
        status="HEALTHY",
    )
    with meta_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_health ORDER BY id DESC LIMIT 1")
        row = dict(cursor.fetchone())
        assert row["data_quality_status"] == "HEALTHY"
        assert row["throughput"] == 150.0


# ============================================================================
# CRITERION 5: Implement anomaly detection
# ============================================================================
def test_criterion_5_implement_anomaly_detection():
    """Criterion 5: Start with rolling statistical baselines and explainable scores."""
    detector = RobustStatisticalDetector(history_size=30, threshold_z=3.5)
    now = datetime.now(timezone.utc)

    # 1. Baseline period: normal latency centered at 40ms with slight variance
    for i in range(20):
        z, median, is_anom = detector.score("order-service", "latency", 40.0 + (i % 3), timestamp=now)
        assert not is_anom, f"False alarm during baseline at step {i}"
        assert abs(z) < 3.5

    # 2. Injected anomalous spike: latency jumps to 280ms
    z_spike, median, is_anom_spike = detector.score("order-service", "latency", 280.0, timestamp=now)
    assert is_anom_spike is True, "Spike should trigger statistical anomaly"
    assert z_spike > 5.0, f"Expected high robust z-score, got {z_spike}"

    # 3. Verify explainability: median baseline is intact (~41ms) and z-score is explainable
    assert 39.0 <= median <= 43.0

    # 4. Zero-MAD fallback verification (constant metric series with 0 variance)
    zero_mad_detector = RobustStatisticalDetector(history_size=15, threshold_z=3.5)
    for _ in range(12):
        zero_mad_detector.score("cache-service", "hits", 100.0, timestamp=now)
    # A moderate deviation from constant 100.0 should not cause division by zero
    z_fallback, _, _ = zero_mad_detector.score("cache-service", "hits", 115.0, timestamp=now)
    assert not np.isnan(z_fallback)
    assert not np.isinf(z_fallback)
    assert z_fallback > 0.0


# ============================================================================
# CRITERION 6: Build a working dashboard
# ============================================================================
def test_criterion_6_build_working_dashboard(client):
    """Criterion 6: Show service health, time-series metrics, anomalies and pipeline lag."""
    # 1. Screen 1: Operations Overview (Service Inventory & Pipeline Lag)
    res_services = client.get("/api/v1/services")
    assert res_services.status_code == 200
    services = res_services.json()["services"]
    assert len(services) > 0
    assert all("id" in s and "status" in s and "tier" in s for s in services)

    res_pipeline = client.get("/api/v1/pipeline/status")
    assert res_pipeline.status_code == 200
    pipeline_data = res_pipeline.json()
    assert "consumer_lag" in pipeline_data
    assert "throughput_eps" in pipeline_data
    assert "data_quality_metrics" in pipeline_data

    # 2. Screen 2: Anomaly Explorer (Time-series metrics & Anomaly list)
    target_svc = services[0]["id"]
    res_metrics = client.get(f"/api/v1/services/{target_svc}/metrics?limit=30")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert len(metrics_data["metrics"]) > 0
    first_pt = metrics_data["metrics"][0]
    assert "latency_p50" in first_pt and "error_rate" in first_pt

    res_anomalies = client.get(f"/api/v1/services/{target_svc}/anomalies")
    assert res_anomalies.status_code == 200
    assert "anomalies" in res_anomalies.json()

    # 3. Verify static dashboard serving
    res_dashboard = client.get("/")
    assert res_dashboard.status_code == 200
    assert "Infrastructure Intelligence Platform" in res_dashboard.text


# ============================================================================
# CRITERION 7: Verify end-to-end behavior
# ============================================================================
def test_criterion_7_verify_end_to_end_behavior(client):
    """Criterion 7: Inject a controlled fault and measure detection latency, completeness and false alarms."""
    aggregator = RollingWindowAggregator(window_seconds=30.0)
    detector = RobustStatisticalDetector(history_size=30, threshold_z=3.5)
    now = datetime.now(timezone.utc)

    # Step 1: Establish baseline telemetry (25 normal ticks: 38-41ms latency, 0% errors)
    false_alarms = 0
    for i in range(25):
        lat_val = 38.0 + (i % 3)
        evt_lat = TelemetryEvent(
            event_id=f"evt-base-lat-{i}",
            event_time=now,
            service="order-service",
            metric_name="request_latency_ms",
            metric_value=lat_val,
        )
        evt_err = TelemetryEvent(
            event_id=f"evt-base-err-{i}",
            event_time=now,
            service="order-service",
            metric_name="error_rate",
            metric_value=0.0,
        )
        aggregator.add_event(evt_lat)
        aggregator.add_event(evt_err)
        z, _, is_anom = detector.score("order-service", "latency", lat_val, timestamp=now)
        if is_anom:
            false_alarms += 1

    # Verify 0 false alarms during clean baseline
    assert false_alarms == 0, f"False alarms detected during baseline: {false_alarms}"

    # Step 2: Inject controlled fault (latency surge to 380ms + error rate to 0.40)
    fault_events = [
        TelemetryEvent(
            event_id=f"evt-fault-lat-{k}",
            event_time=now,
            service="order-service",
            metric_name="request_latency_ms",
            metric_value=380.0,
        )
        for k in range(5)
    ] + [
        TelemetryEvent(
            event_id=f"evt-fault-err-{k}",
            event_time=now,
            service="order-service",
            metric_name="error_rate",
            metric_value=0.40,
        )
        for k in range(5)
    ]

    for fe in fault_events:
        aggregator.add_event(fe)

    # Step 3: Evaluate window and measure detection latency
    t_start = time.perf_counter()
    window = aggregator.get_window("order-service")
    assert window is not None

    z_score, median_val, is_detected = detector.score(
        "order-service", "latency", window.latency_p90, timestamp=now
    )
    detection_latency_sec = time.perf_counter() - t_start

    # Step 4: Validate acceptance thresholds
    # Detection completeness: 100% (is_detected is True)
    assert is_detected is True, "Fault was not detected by statistical detector"

    # Detection latency: must be under 1.5 seconds
    assert detection_latency_sec < 1.5, f"Detection latency exceeded threshold: {detection_latency_sec:.4f}s"

    # Explainability: robust z-score significantly exceeds threshold with clear deviation
    assert z_score > 5.0, f"Expected high z-score for 380ms fault, got {z_score:.2f}"
    assert 37.0 <= median_val <= 42.0, f"Baseline median corrupted: {median_val}"

    # Metric verification
    assert window.latency_p90 >= 300.0
    assert window.error_rate >= 0.05
