from datetime import datetime, timezone, timedelta
import pytest
from pydantic import ValidationError
from storage.models import TelemetryEvent
from services.ingestion.collector import TelemetryCollector

def test_telemetry_schema_valid():
    evt = TelemetryEvent(
        event_id="evt-dq-1",
        event_time=datetime.now(timezone.utc),
        service="checkout-api",
        environment="production",
        event_type="service_metric",
        metric_name="request_latency_ms",
        metric_value=45.2,
        unit="ms"
    )
    assert evt.service == "checkout-api"
    assert evt.metric_value == 45.2


def test_banned_unbounded_attribute_rejected():
    collector = TelemetryCollector()
    raw = {
        "event_id": "evt-banned",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "service": "checkout-api",
        "metric_name": "latency",
        "metric_value": 40.0,
        "attributes": {"user_id": "cust-9999"}
    }
    event, err = collector.validate_event(raw)
    assert event is None
    assert "banned" in err.lower()
