from datetime import datetime, timezone
import math
import random
from typing import Any, Dict, List, Optional
import uuid
from storage.models import TelemetryEvent


class TelemetrySimulator:
    """Simulates multi-tier microservice architecture telemetry with diurnal patterns."""

    def __init__(self):
        self.services = [
            "edge-gateway",
            "auth-service",
            "catalog-service",
            "checkout-api",
            "order-service",
            "payment-service",
            "inventory-service",
            "notification-service",
        ]
        self.tick_counter = 0

    def generate_tick(self, active_fault: Optional[Dict[str, Any]] = None) -> List[TelemetryEvent]:
        self.tick_counter += 1
        now = datetime.now(timezone.utc)
        events: List[TelemetryEvent] = []

        # Diurnal wave
        hour = (self.tick_counter / 60.0) % 24
        diurnal_factor = 1.0 + 0.4 * math.sin((hour - 8.0) * math.pi / 12.0)

        for svc in self.services:
            base_req = 120.0 * diurnal_factor
            base_lat = 45.0
            base_err = 0.002
            base_cpu = 30.0

            if active_fault and active_fault.get("service") == svc:
                scenario = active_fault.get("scenario")
                if scenario == "payment_cascade":
                    base_lat = 250.0
                    base_err = 0.18
                elif scenario == "inventory_cpu_exhaustion":
                    base_cpu = 92.0
                    base_lat = 180.0
                elif scenario == "gateway_traffic_surge":
                    base_req = 450.0

            # 1. Latency metric
            lat_val = max(5.0, random.gauss(base_lat, base_lat * 0.15))
            events.append(TelemetryEvent(
                event_id=f"sim-{uuid.uuid4().hex[:10]}",
                event_time=now,
                service=svc,
                environment="production",
                event_type="service_metric",
                metric_name="request_latency_ms",
                metric_value=round(lat_val, 2),
                unit="ms",
            ))

            # 2. Error rate metric
            err_val = max(0.0, min(1.0, random.gauss(base_err, base_err * 0.2)))
            events.append(TelemetryEvent(
                event_id=f"sim-{uuid.uuid4().hex[:10]}",
                event_time=now,
                service=svc,
                environment="production",
                event_type="service_metric",
                metric_name="error_rate",
                metric_value=round(err_val, 4),
                unit="ratio",
            ))

            # 3. CPU utilization
            cpu_val = max(5.0, min(100.0, random.gauss(base_cpu, 4.0)))
            events.append(TelemetryEvent(
                event_id=f"sim-{uuid.uuid4().hex[:10]}",
                event_time=now,
                service=svc,
                environment="production",
                event_type="service_metric",
                metric_name="cpu_utilization",
                metric_value=round(cpu_val, 2),
                unit="percent",
            ))

        return events
