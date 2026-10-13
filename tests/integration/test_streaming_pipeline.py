from datetime import datetime, timezone
import pytest
from services.feature_engine.rolling_windows import RollingWindowAggregator
from storage.broker import EmbeddedEventBroker
from storage.lakehouse import LakehouseManager
from storage.models import TelemetryEvent

def test_streaming_ingestion_and_rolling_windows(tmp_path):
    broker = EmbeddedEventBroker(storage_dir=tmp_path / "broker", num_partitions=2)
    lakehouse = LakehouseManager(lakehouse_dir=tmp_path / "lakehouse")
    windows = RollingWindowAggregator(window_seconds=60.0)

    events = [
        TelemetryEvent(
            event_id=f"evt-{i}",
            event_time=datetime.now(timezone.utc),
            service="catalog-service",
            metric_name="request_latency_ms",
            metric_value=30.0 + i,
            unit="ms",
        )
        for i in range(10)
    ]

    for evt in events:
        broker.publish("telemetry.events", evt)
        windows.add_event(evt)

    lakehouse.write_events(events)
    assert lakehouse.total_records == 10

    # Poll broker
    polled = broker.poll("telemetry.events", "group_test", max_records=20)
    assert len(polled) == 10

    # Window aggregation
    win = windows.get_window("catalog-service")
    assert win is not None
    assert win.sample_count == 10
    assert win.latency_p50 > 30.0
