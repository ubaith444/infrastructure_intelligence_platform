from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid
from storage.models import TelemetryEvent


def parse_otlp_metrics(payload: Dict[str, Any]) -> List[TelemetryEvent]:
    events: List[TelemetryEvent] = []
    now = datetime.now(timezone.utc)
    for rm in payload.get("resourceMetrics", []):
        service_name = "unknown_service"
        for attr in rm.get("resource", {}).get("attributes", []):
            if attr.get("key") == "service.name":
                service_name = attr.get("value", {}).get("stringValue", service_name)

        for sm in rm.get("scopeMetrics", []):
            for m in sm.get("metrics", []):
                metric_name = m.get("name", "generic_metric")
                gauge_data = m.get("gauge", {}).get("dataPoints", [])
                sum_data = m.get("sum", {}).get("dataPoints", [])
                for dp in gauge_data + sum_data:
                    val = dp.get("asDouble", dp.get("asInt", 0.0))
                    events.append(TelemetryEvent(
                        event_id=f"otlp-{uuid.uuid4().hex[:10]}",
                        event_time=now,
                        service=service_name,
                        environment="production",
                        event_type="service_metric",
                        metric_name=metric_name,
                        metric_value=float(val),
                        unit=m.get("unit", ""),
                    ))
    return events
