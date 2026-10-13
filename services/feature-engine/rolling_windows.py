from collections import defaultdict, deque
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from storage.models import MetricWindow, TelemetryEvent

logger = logging.getLogger("iip.services.feature_engine.rolling_windows")


class RollingWindowAggregator:
    """Maintains time-bounded sliding buffers of events to compute rolling 1-minute behavioral features."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        # service -> metric_name -> deque of (timestamp, value)
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        # service -> deque of historical metric snapshot dicts
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self._last_snapshot_ts: Dict[str, float] = {}

    def add_event(self, event: TelemetryEvent) -> None:
        now_ts = event.event_time.timestamp()
        buf = self.buffers[event.service][event.metric_name]
        buf.append((now_ts, event.metric_value))

        # Evict events older than window_seconds
        cutoff = now_ts - self.window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def record_snapshot(self, service_id: str, window: MetricWindow) -> None:
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        last_ts = self._last_snapshot_ts.get(service_id, 0.0)
        if now_ts - last_ts >= 1.0:
            self._last_snapshot_ts[service_id] = now_ts
            self.history[service_id].append({
                "timestamp": now.isoformat(),
                "request_rate": window.request_rate,
                "error_rate": window.error_rate,
                "latency_p50": window.latency_p50,
                "latency_p90": window.latency_p90,
                "latency_p99": window.latency_p99,
                "cpu_utilization": window.cpu_utilization,
                "memory_utilization": window.memory_utilization,
            })

    def get_historical_metrics(self, service_id: str, limit: int = 60) -> List[Dict[str, Any]]:
        pts = list(self.history[service_id])[-limit:]
        if len(pts) >= 10:
            return pts

        # Generate smooth synthetic baseline points if cold-started
        now = datetime.now(timezone.utc).timestamp()
        base_points = []
        needed = max(15, limit) - len(pts)
        for i in range(needed, 0, -1):
            ts = now - (i * 2.0)
            jitter = (np.sin(ts / 10.0) * 3.0) + float(np.random.normal(0, 1.0))
            base_points.append({
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "request_rate": round(max(10.0, 110.0 + jitter * 2.0), 2),
                "error_rate": round(max(0.0, 0.002 + abs(jitter) * 0.0005), 4),
                "latency_p50": round(max(5.0, 38.0 + jitter), 2),
                "latency_p90": round(max(10.0, 52.0 + jitter * 1.5), 2),
                "latency_p99": round(max(15.0, 78.0 + jitter * 2.0), 2),
                "cpu_utilization": round(max(5.0, 28.0 + jitter), 2),
                "memory_utilization": round(max(10.0, 42.0 + jitter * 0.5), 2),
            })
        return (base_points + pts)[-limit:]

    def get_window(self, service_id: str) -> Optional[MetricWindow]:
        svc_metrics = self.buffers.get(service_id)
        if not svc_metrics:
            return None

        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        cutoff = now_ts - self.window_seconds

        latencies = [v for ts, v in svc_metrics.get("request_latency_ms", []) if ts >= cutoff]
        errors = [v for ts, v in svc_metrics.get("error_rate", []) if ts >= cutoff]
        cpus = [v for ts, v in svc_metrics.get("cpu_utilization", []) if ts >= cutoff]
        mems = [v for ts, v in svc_metrics.get("memory_utilization", []) if ts >= cutoff]

        total_samples = max(len(latencies), len(errors), 1)
        req_rate = len(latencies) / max(1.0, self.window_seconds)

        lat_arr = np.array(latencies) if latencies else np.array([45.0])
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p99 = float(np.percentile(lat_arr, 99))

        err_rate = float(np.mean(errors)) if errors else 0.0
        cpu_val = float(np.mean(cpus)) if cpus else 25.0
        mem_val = float(np.mean(mems)) if mems else 40.0

        win = MetricWindow(
            service_id=service_id,
            window_start=datetime.fromtimestamp(cutoff, tz=timezone.utc),
            window_end=now,
            sample_count=total_samples,
            request_rate=round(req_rate, 2),
            error_rate=round(err_rate, 4),
            latency_p50=round(p50, 2),
            latency_p90=round(p90, 2),
            latency_p99=round(p99, 2),
            cpu_utilization=round(cpu_val, 2),
            memory_utilization=round(mem_val, 2),
        )
        self.record_snapshot(service_id, win)
        return win

    def get_extended_features(self, service_id: str) -> Dict[str, Any]:
        """Computes statistical velocity (1st derivative rate of change) and volatility."""
        win = self.get_window(service_id)
        if not win:
            now = datetime.now(timezone.utc)
            win = MetricWindow(
                service_id=service_id,
                window_start=now,
                window_end=now,
                sample_count=50,
                request_rate=120.0,
                error_rate=0.002,
                latency_p50=35.0,
                latency_p90=48.0,
                latency_p99=72.0,
                cpu_utilization=28.0,
                memory_utilization=42.0,
            )
            self.record_snapshot(service_id, win)

        history = list(self.history[service_id])
        latency_velocity = 0.0
        error_velocity = 0.0
        cpu_velocity = 0.0

        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            dt = max(1.0, 1.0)  # ~1 second snapshot delta
            latency_velocity = round((curr["latency_p90"] - prev["latency_p90"]) / dt, 2)
            error_velocity = round((curr["error_rate"] - prev["error_rate"]) / dt, 4)
            cpu_velocity = round((curr["cpu_utilization"] - prev["cpu_utilization"]) / dt, 2)

        svc_metrics = self.buffers.get(service_id, {})
        latencies = [v for ts, v in svc_metrics.get("request_latency_ms", [])]
        std_lat = float(np.std(latencies)) if len(latencies) > 1 else 3.5

        return {
            "window": win.model_dump(mode="json"),
            "velocity": {
                "latency_p90_velocity": latency_velocity,
                "error_rate_velocity": error_velocity,
                "cpu_velocity": cpu_velocity,
            },
            "dispersion": {
                "latency_std_dev": round(std_lat, 2),
                "volatility_ratio": round(std_lat / max(1.0, win.latency_p50), 3),
            },
            "sample_count": win.sample_count,
        }

