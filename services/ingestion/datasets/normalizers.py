from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from storage.models import TelemetryEvent


class GoogleClusterNormalizer:
    """Normalizes Google Cluster Workload Traces into standard TelemetryEvent metrics."""

    @staticmethod
    def normalize(record: Dict[str, Any]) -> List[TelemetryEvent]:
        events = []
        ts_millis = record.get("timestamp_millis", int(datetime.now(timezone.utc).timestamp() * 1000))
        event_time = datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc)
        machine_id = str(record.get("machine_id", "google-worker-node"))
        job_id = record.get("job_id", "job-unknown")

        common_attrs = {
            "source_dataset": "GoogleClusterWorkload",
            "job_id": str(job_id),
            "task_index": str(record.get("task_index", 0)),
            "task_event_type": str(record.get("task_event_type", "RUNNING")),
            "priority": str(record.get("priority", 0)),
        }

        # 1. CPU metric
        cpu_val = float(record.get("cpu_rate", 0.0)) * 100.0  # normalize to 0-100%
        events.append(
            TelemetryEvent(
                event_id=f"gcp-cpu-{uuid.uuid4().hex[:10]}",
                event_time=event_time,
                service="cluster-scheduler",
                environment="production",
                event_type="system_metric",
                metric_name="cpu_utilization",
                metric_value=round(cpu_val, 2),
                unit="percent",
                host_id=machine_id,
                attributes=common_attrs,
            )
        )

        # 2. Canonical memory usage
        mem_val = float(record.get("canonical_memory_usage", 0.0)) * 100.0
        events.append(
            TelemetryEvent(
                event_id=f"gcp-mem-{uuid.uuid4().hex[:10]}",
                event_time=event_time,
                service="cluster-scheduler",
                environment="production",
                event_type="system_metric",
                metric_name="memory_utilization",
                metric_value=round(mem_val, 2),
                unit="percent",
                host_id=machine_id,
                attributes=common_attrs,
            )
        )

        # 3. Disk I/O time
        if "disk_io_time" in record:
            events.append(
                TelemetryEvent(
                    event_id=f"gcp-io-{uuid.uuid4().hex[:10]}",
                    event_time=event_time,
                    service="cluster-scheduler",
                    environment="production",
                    event_type="system_metric",
                    metric_name="disk_io_time_ratio",
                    metric_value=round(float(record["disk_io_time"]), 4),
                    unit="ratio",
                    host_id=machine_id,
                    attributes=common_attrs,
                )
            )

        return events


class LogHubNormalizer:
    """Normalizes LogHub distributed logs (HDFS / Hadoop) into TelemetryEvents."""

    @staticmethod
    def normalize(record: Dict[str, Any]) -> TelemetryEvent:
        now = datetime.now(timezone.utc)
        date_str = record.get("date")
        time_str = record.get("time")
        if date_str and time_str:
            try:
                event_time = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
            except Exception:
                event_time = now
        else:
            event_time = now

        level = str(record.get("level", "INFO")).upper()
        # Metric weight: INFO=1.0, WARN=2.0, ERROR=3.0, FATAL=4.0
        weight_map = {"INFO": 1.0, "WARN": 2.0, "ERROR": 3.0, "FATAL": 4.0}
        metric_val = weight_map.get(level, 1.0)
        is_anomaly = 1.0 if str(record.get("anomaly_label", "Normal")).lower() != "normal" else 0.0

        attrs = {
            "source_dataset": "LogHub_HDFS",
            "log_level": level,
            "component": str(record.get("component", "HDFS")),
            "event_template": str(record.get("event_template", "")),
            "block_id": str(record.get("block_id", "")),
            "anomaly_label": str(record.get("anomaly_label", "Normal")),
        }

        return TelemetryEvent(
            event_id=f"loghub-{uuid.uuid4().hex[:10]}",
            event_time=event_time,
            service="distributed-storage",
            environment="production",
            event_type="log_event",
            metric_name="log_severity_weight",
            metric_value=metric_val if is_anomaly == 0.0 else 5.0,
            unit="severity_index",
            host_id="namenode-01",
            attributes=attrs,
        )


class OpenTelemetryTraceNormalizer:
    """Normalizes OpenTelemetry distributed traces and span records into TelemetryEvents."""

    @staticmethod
    def normalize(record: Dict[str, Any]) -> TelemetryEvent:
        now = datetime.now(timezone.utc)
        start_nano = record.get("start_time_unix_nano")
        if start_nano:
            event_time = datetime.fromtimestamp(start_nano / 1e9, tz=timezone.utc)
        else:
            event_time = now

        service = str(record.get("service_name", "order-service"))
        dur_ms = float(record.get("duration_millis", 45.0))
        trace_id = str(record.get("trace_id", uuid.uuid4().hex))
        span_id = str(record.get("span_id", uuid.uuid4().hex[:16]))

        attrs = {
            "source_dataset": "OpenTelemetry_Demo",
            "operation": str(record.get("operation_name", "unknown")),
            "parent_span_id": str(record.get("parent_span_id") or ""),
            "status_code": str(record.get("status_code", "OK")),
        }

        return TelemetryEvent(
            event_id=f"otel-span-{uuid.uuid4().hex[:10]}",
            event_time=event_time,
            service=service,
            environment="production",
            event_type="trace_span",
            metric_name="request_latency_ms",
            metric_value=round(dur_ms, 2),
            unit="ms",
            trace_id=trace_id,
            span_id=span_id,
            attributes=attrs,
        )
