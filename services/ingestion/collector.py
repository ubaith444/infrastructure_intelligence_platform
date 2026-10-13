from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
from storage.dlq import DeadLetterQueueManager
from storage.models import LateEventPolicy, TelemetryEvent

logger = logging.getLogger("iip.services.ingestion.collector")


class TelemetryCollector:
    """Validates incoming telemetry events, enforces bounds, manages event-time watermarking,

    and isolates violations and late-arriving events to DLQ.
    """

    def __init__(
        self,
        dlq_manager: Optional[DeadLetterQueueManager] = None,
        allowed_lateness_seconds: float = 30.0,
        late_policy: LateEventPolicy = LateEventPolicy.ROUTE_TO_DLQ,
    ):
        self.dlq = dlq_manager or DeadLetterQueueManager()
        self.processed_count = 0
        self.rejected_count = 0
        self.late_events_dropped = 0
        self.late_events_quarantined = 0

        self.allowed_lateness_seconds = allowed_lateness_seconds
        self.late_policy = late_policy
        self.max_event_time_ts = 0.0
        self.watermark_ts = 0.0

    def validate_event(self, event_data: Any) -> Tuple[Optional[TelemetryEvent], Optional[str]]:
        if isinstance(event_data, dict):
            # Check label cardinality
            attrs = event_data.get("attributes", {})
            if len(attrs) > 25:
                return None, "Attributes cardinality exceeded limit (>25 tags)"
            # Check for banned unbounded keys
            banned_keys = {"user_id", "email", "raw_url", "token"}
            for k in attrs:
                if k.lower() in banned_keys:
                    return None, f"Banned unbounded attribute key detected: {k}"

            try:
                event = TelemetryEvent(**event_data)
                return event, None
            except Exception as e:
                return None, str(e)
        elif isinstance(event_data, TelemetryEvent):
            return event_data, None
        return None, "Invalid event type"

    def update_watermark(self, event_time_ts: float) -> None:
        """Advances the event-time watermark monotonically."""
        if event_time_ts > self.max_event_time_ts:
            self.max_event_time_ts = event_time_ts
            self.watermark_ts = max(0.0, self.max_event_time_ts - self.allowed_lateness_seconds)

    def is_late(self, event_time_ts: float) -> bool:
        """Determines if an event arrived behind the active watermark."""
        if self.watermark_ts <= 0.0:
            return False
        return event_time_ts < self.watermark_ts

    def ingest(self, raw_events: List[Any]) -> List[TelemetryEvent]:
        valid_events = []
        for raw in raw_events:
            evt, err = self.validate_event(raw)
            if not evt:
                self.rejected_count += 1
                self.dlq.record_violation(raw, err or "Unknown validation error")
                continue

            evt_ts = evt.event_time.timestamp()

            # Watermark and late-arrival evaluation
            if self.is_late(evt_ts):
                if self.late_policy == LateEventPolicy.DROP:
                    self.late_events_dropped += 1
                    logger.debug(f"Dropped late-arriving event {evt.event_id}")
                    continue
                elif self.late_policy == LateEventPolicy.ROUTE_TO_DLQ:
                    self.late_events_quarantined += 1
                    self.dlq.record_violation(
                        evt.model_dump(mode="json"),
                        f"Late event arrived behind watermark (event_time={evt.event_time.isoformat()}, watermark={datetime.fromtimestamp(self.watermark_ts, tz=timezone.utc).isoformat()})",
                    )
                    continue

            # Event is accepted: update watermark
            self.update_watermark(evt_ts)
            valid_events.append(evt)
            self.processed_count += 1

        return valid_events

    def get_watermark_status(self) -> Dict[str, Any]:
        now_ts = datetime.now(timezone.utc).timestamp()
        delay_ms = round(max(0.0, (now_ts - self.watermark_ts) * 1000.0), 1) if self.watermark_ts > 0 else 0.0
        return {
            "current_watermark": datetime.fromtimestamp(self.watermark_ts, tz=timezone.utc).isoformat() if self.watermark_ts > 0 else None,
            "max_event_time": datetime.fromtimestamp(self.max_event_time_ts, tz=timezone.utc).isoformat() if self.max_event_time_ts > 0 else None,
            "allowed_lateness_seconds": self.allowed_lateness_seconds,
            "watermark_delay_ms": delay_ms,
            "late_policy": self.late_policy.value,
            "late_events_dropped": self.late_events_dropped,
            "late_events_quarantined": self.late_events_quarantined,
        }

    @staticmethod
    def parse_otlp_payload(payload: Any) -> List[Dict[str, Any]]:
        return parse_otlp_payload(payload)


def _extract_otlp_attr_value(val: Any) -> Any:
    if isinstance(val, dict):
        if "stringValue" in val:
            return val["stringValue"]
        if "intValue" in val:
            return int(val["intValue"])
        if "doubleValue" in val:
            return float(val["doubleValue"])
        if "boolValue" in val:
            return bool(val["boolValue"])
    return val


def _extract_otlp_attrs(attr_list: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not isinstance(attr_list, list):
        return result
    for item in attr_list:
        if isinstance(item, dict) and "key" in item:
            result[item["key"]] = _extract_otlp_attr_value(item.get("value"))
    return result


def parse_otlp_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalizes OpenTelemetry OTLP JSON (resourceMetrics, resourceSpans) or raw event dicts

    into standard platform TelemetryEvent dictionary representations.
    """
    if not payload:
        return []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if "events" in payload and isinstance(payload["events"], list):
        return payload["events"]

    events: List[Dict[str, Any]] = []

    # 1. Parse OTLP Resource Metrics
    if "resourceMetrics" in payload and isinstance(payload["resourceMetrics"], list):
        for rm in payload["resourceMetrics"]:
            res = rm.get("resource", {})
            res_attrs = _extract_otlp_attrs(res.get("attributes", []))
            service_name = res_attrs.get("service.name", "unknown-service")
            env = res_attrs.get("deployment.environment", "production")

            for sm in rm.get("scopeMetrics", []):
                for metric in sm.get("metrics", []):
                    metric_name = metric.get("name", "unnamed_metric")
                    unit = metric.get("unit")
                    dp_sources = []
                    for k in ("gauge", "sum", "histogram", "summary"):
                        if k in metric and isinstance(metric[k], dict) and "dataPoints" in metric[k]:
                            dp_sources.extend(metric[k]["dataPoints"])

                    for dp in dp_sources:
                        val = dp.get("asDouble")
                        if val is None:
                            val = dp.get("asInt")
                        if val is None and "value" in dp:
                            val = dp.get("value")
                        if val is None:
                            val = 0.0

                        time_nano = dp.get("timeUnixNano")
                        if time_nano:
                            try:
                                dt = datetime.fromtimestamp(int(time_nano) / 1e9, tz=timezone.utc)
                            except Exception:
                                dt = datetime.now(timezone.utc)
                        else:
                            dt = datetime.now(timezone.utc)

                        dp_attrs = _extract_otlp_attrs(dp.get("attributes", []))
                        merged_attrs = {**res_attrs, **dp_attrs}
                        merged_attrs.pop("service.name", None)

                        events.append({
                            "event_id": f"evt-otlp-{uuid.uuid4().hex[:12]}",
                            "event_time": dt.isoformat(),
                            "service": service_name,
                            "environment": env,
                            "event_type": "service_metric",
                            "metric_name": metric_name,
                            "metric_value": float(val),
                            "unit": unit,
                            "attributes": merged_attrs,
                        })

    # 2. Parse OTLP Resource Spans
    if "resourceSpans" in payload and isinstance(payload["resourceSpans"], list):
        for rs in payload["resourceSpans"]:
            res = rs.get("resource", {})
            res_attrs = _extract_otlp_attrs(res.get("attributes", []))
            service_name = res_attrs.get("service.name", "unknown-service")
            env = res_attrs.get("deployment.environment", "production")

            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    trace_id = span.get("traceId", "")
                    span_id = span.get("spanId", "")
                    start_nano = int(span.get("startTimeUnixNano", 0))
                    end_nano = int(span.get("endTimeUnixNano", 0))
                    dur_ms = max(0.01, (end_nano - start_nano) / 1e6) if end_nano > start_nano else 1.0

                    status = span.get("status", {})
                    code = status.get("code")
                    is_err = 1.0 if (code in (2, "STATUS_CODE_ERROR", "ERROR") or status.get("message")) else 0.0

                    if start_nano > 0:
                        try:
                            dt = datetime.fromtimestamp(start_nano / 1e9, tz=timezone.utc)
                        except Exception:
                            dt = datetime.now(timezone.utc)
                    else:
                        dt = datetime.now(timezone.utc)

                    span_attrs = _extract_otlp_attrs(span.get("attributes", []))
                    merged_attrs = {**res_attrs, **span_attrs}
                    merged_attrs.pop("service.name", None)

                    events.append({
                        "event_id": f"evt-otlp-{uuid.uuid4().hex[:12]}",
                        "event_time": dt.isoformat(),
                        "service": service_name,
                        "environment": env,
                        "event_type": "service_metric",
                        "metric_name": "request_latency_ms",
                        "metric_value": round(dur_ms, 2),
                        "unit": "ms",
                        "trace_id": trace_id or None,
                        "span_id": span_id or None,
                        "attributes": merged_attrs,
                    })

                    events.append({
                        "event_id": f"evt-otlp-{uuid.uuid4().hex[:12]}",
                        "event_time": dt.isoformat(),
                        "service": service_name,
                        "environment": env,
                        "event_type": "service_metric",
                        "metric_name": "error_rate",
                        "metric_value": is_err,
                        "unit": "ratio",
                        "trace_id": trace_id or None,
                        "span_id": span_id or None,
                        "attributes": merged_attrs,
                    })

    if not events and isinstance(payload, dict) and "metric_name" in payload and "service" in payload:
        return [payload]

    return events
