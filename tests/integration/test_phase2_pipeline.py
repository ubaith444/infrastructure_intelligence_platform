import asyncio
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import pytest
from apps.api.server import app
from services.feature_engine.rolling_windows import RollingWindowAggregator
from services.ingestion.collector import TelemetryCollector
from services.ingestion.datasets.dataset_loader import DatasetLoader
from services.ingestion.datasets.normalizers import (
    GoogleClusterNormalizer,
    LogHubNormalizer,
    OpenTelemetryTraceNormalizer,
)
from storage.broker import EmbeddedEventBroker
from storage.models import LateEventPolicy, TelemetryEvent


@pytest.fixture
def client():
    return TestClient(app)


def test_durable_broker_offset_commit_and_seek(tmp_path):
    broker = EmbeddedEventBroker(storage_dir=tmp_path / "broker", num_partitions=2)

    now = datetime.now(timezone.utc)
    evts = [
        TelemetryEvent(
            event_id=f"broker-test-{i}",
            event_time=now,
            service="checkout-api",
            metric_name="request_latency_ms",
            metric_value=40.0 + i,
        )
        for i in range(10)
    ]

    for e in evts:
        broker.publish("telemetry.test", e)

    # Poll first 5
    records = broker.poll("telemetry.test", "worker_cg", max_records=5, auto_commit=True)
    assert len(records) == 5

    # Check lag
    lag = broker.get_lag("telemetry.test", "worker_cg")
    assert lag["total_lag"] == 5

    # Seek back to 0
    broker.reset_offsets("telemetry.test", "worker_cg", target="earliest")
    lag_reset = broker.get_lag("telemetry.test", "worker_cg")
    assert lag_reset["total_lag"] == 10

    # Poll again from offset 0
    replayed = broker.poll("telemetry.test", "worker_cg", max_records=10, auto_commit=True)
    assert len(replayed) == 10


def test_broker_idempotence_and_deduplication(tmp_path):
    broker = EmbeddedEventBroker(storage_dir=tmp_path / "broker_dedup", num_partitions=2)
    now = datetime.now(timezone.utc)
    evt = TelemetryEvent(
        event_id="evt-unique-dup-1",
        event_time=now,
        service="payment-service",
        metric_name="error_rate",
        metric_value=0.0,
    )

    pid1, off1, is_dup1 = broker.publish("telemetry.events", evt)
    assert is_dup1 is False
    assert off1 >= 0

    # Publish duplicate event_id
    pid2, off2, is_dup2 = broker.publish("telemetry.events", evt)
    assert is_dup2 is True
    assert off2 == -1


def test_event_time_watermark_and_late_arrival():
    collector = TelemetryCollector(
        allowed_lateness_seconds=20.0,
        late_policy=LateEventPolicy.ROUTE_TO_DLQ,
    )
    now = datetime.now(timezone.utc)

    # Event 1: arrives at T0
    e1 = TelemetryEvent(
        event_id="evt-on-time-1",
        event_time=now,
        service="order-service",
        metric_name="latency",
        metric_value=30.0,
    )
    res1 = collector.ingest([e1])
    assert len(res1) == 1

    # Event 2: arrives at T0 + 60s -> advances watermark to (T0 + 40s)
    e2 = TelemetryEvent(
        event_id="evt-advanced-2",
        event_time=now + timedelta(seconds=60),
        service="order-service",
        metric_name="latency",
        metric_value=35.0,
    )
    res2 = collector.ingest([e2])
    assert len(res2) == 1

    # Event 3: arrives with timestamp T0 (which is < watermark T0 + 40s) -> late!
    e3 = TelemetryEvent(
        event_id="evt-late-3",
        event_time=now + timedelta(seconds=10),
        service="order-service",
        metric_name="latency",
        metric_value=99.0,
    )
    res3 = collector.ingest([e3])
    # Should be rejected / quarantined to DLQ because it is late!
    assert len(res3) == 0
    assert collector.late_events_quarantined == 1


def test_extended_rolling_features_velocity():
    aggregator = RollingWindowAggregator(window_seconds=60.0)
    now = datetime.now(timezone.utc)

    for i in range(10):
        evt = TelemetryEvent(
            event_id=f"feat-{i}",
            event_time=now + timedelta(seconds=i),
            service="checkout-api",
            metric_name="request_latency_ms",
            metric_value=40.0 + (i * 5.0),
        )
        aggregator.add_event(evt)

    features = aggregator.get_extended_features("checkout-api")
    assert "window" in features
    assert "velocity" in features
    assert "dispersion" in features
    assert features["sample_count"] == 10


def test_public_dataset_normalizers():
    # 1. Google Cluster Normalizer
    g_raw = {
        "timestamp_millis": 1776000000000,
        "job_id": 100,
        "task_index": 0,
        "machine_id": "m1",
        "cpu_rate": 0.85,
        "canonical_memory_usage": 0.70,
        "disk_io_time": 0.05,
    }
    g_events = GoogleClusterNormalizer.normalize(g_raw)
    assert len(g_events) == 3
    assert any(e.metric_name == "cpu_utilization" and e.metric_value == 85.0 for e in g_events)
    assert any(e.metric_name == "memory_utilization" and e.metric_value == 70.0 for e in g_events)

    # 2. LogHub Normalizer
    l_raw = {
        "date": "2026-10-12",
        "time": "14:00:00",
        "level": "ERROR",
        "component": "HDFS",
        "content": "Disk full error",
        "event_template": "Disk full <*>",
        "anomaly_label": "Anomaly",
    }
    l_evt = LogHubNormalizer.normalize(l_raw)
    assert l_evt.event_type == "log_event"
    assert l_evt.metric_value == 5.0  # anomaly weight

    # 3. OpenTelemetry Trace Normalizer
    otel_raw = {
        "trace_id": "trace-123",
        "span_id": "span-456",
        "parent_span_id": "span-parent",
        "service_name": "checkout-api",
        "operation_name": "create_order",
        "duration_millis": 55.4,
    }
    otel_evt = OpenTelemetryTraceNormalizer.normalize(otel_raw)
    assert otel_evt.trace_id == "trace-123"
    assert otel_evt.metric_value == 55.4
    assert otel_evt.event_type == "trace_span"


def test_v1_pipeline_datasets_and_replay_api(client):
    # List datasets
    res_ds = client.get("/api/v1/pipeline/datasets")
    assert res_ds.status_code == 200
    data_ds = res_ds.json()
    assert len(data_ds["datasets"]) >= 3
    dataset_ids = [d["id"] for d in data_ds["datasets"]]
    assert "google_cluster" in dataset_ids
    assert "loghub" in dataset_ids
    assert "opentelemetry" in dataset_ids

    # Load dataset
    res_load = client.post("/api/v1/pipeline/datasets/load", json={"dataset_id": "google_cluster"})
    assert res_load.status_code == 200
    assert res_load.json()["status"] == "DATASET_LOADED"
    assert res_load.json()["events_loaded"] > 0

    # Start Replay
    res_replay = client.post(
        "/api/v1/pipeline/replay/start",
        json={"dataset_or_source": "google_cluster", "speed_factor": 10.0, "target_topic": "telemetry.replay"},
    )
    assert res_replay.status_code == 200
    assert res_replay.json()["status"] in ["RUNNING", "COMPLETED"]

    # Replay status
    res_stat = client.get("/api/v1/pipeline/replay/status")
    assert res_stat.status_code == 200
    assert "status" in res_stat.json()

    # Pause Replay
    res_pause = client.post("/api/v1/pipeline/replay/pause")
    assert res_pause.status_code == 200

    # Stop Replay
    res_stop = client.post("/api/v1/pipeline/replay/stop")
    assert res_stop.status_code == 200


def test_v1_dlq_endpoints(client):
    # Query DLQ
    res = client.get("/api/v1/pipeline/dlq")
    assert res.status_code == 200
    data = res.json()
    assert "total_violations" in data
    assert "violations" in data

    # Clear DLQ
    res_clear = client.post("/api/v1/pipeline/dlq/clear")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "DLQ_CLEARED"


def test_v1_service_extended_features(client):
    res = client.get("/api/v1/services/checkout-api/features")
    assert res.status_code == 200
    data = res.json()
    assert data["service_id"] == "checkout-api"
    assert "features" in data
    assert "velocity" in data["features"]
    assert "latency_p90_velocity" in data["features"]["velocity"]
