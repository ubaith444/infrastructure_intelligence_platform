from datetime import datetime, timezone
from typing import Any, Dict, Optional


class FaultInjectionHarness:
    """Manages controlled chaos injection scenarios with ground truth tracking."""

    def __init__(self):
        self.active_fault: Optional[Dict[str, Any]] = None

    def inject_scenario(self, scenario_key: str) -> Dict[str, Any]:
        mapping = {
            "payment_cascade": {"service": "payment-service", "description": "Payment authorization timeout cascade"},
            "inventory_cpu_exhaustion": {"service": "inventory-service", "description": "Database threadpool exhaustion"},
            "gateway_traffic_surge": {"service": "edge-gateway", "description": "Edge volumetric DDoS surge"},
        }

        cfg = mapping.get(scenario_key, {"service": "payment-service", "description": "Custom failure"})
        self.active_fault = {
            "scenario": scenario_key,
            "service": cfg["service"],
            "description": cfg["description"],
            "injected_at": datetime.now(timezone.utc).isoformat(),
            "active": True,
        }
        return self.active_fault

    def clear(self) -> None:
        self.active_fault = None

    def get_status(self) -> Dict[str, Any]:
        return self.active_fault or {"active": False}
