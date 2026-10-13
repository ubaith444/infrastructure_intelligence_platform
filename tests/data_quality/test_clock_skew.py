from datetime import datetime, timezone, timedelta
import pytest
from pydantic import ValidationError
from storage.models import TelemetryEvent

def test_clock_skew_guard():
    future_time = datetime.now(timezone.utc) + timedelta(minutes=25)
    with pytest.raises(ValidationError) as exc_info:
        TelemetryEvent(
            event_id="evt-skew",
            event_time=future_time,
            service="edge-gateway",
            metric_name="cpu",
            metric_value=40.0
        )
    assert "Clock skew detected" in str(exc_info.value)
