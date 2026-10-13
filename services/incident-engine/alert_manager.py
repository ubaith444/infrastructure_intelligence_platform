from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from services.incident_engine.graph import DependencyGraphEngine
from storage.metadata_store import OperationalMetadataStore
from storage.models import AlertRecord, AlertSeverity, AlertStatus, AnomalyRecord


class AlertManager:
    """Manages operational alerts with symptom suppression, 10% alert budget quota, and relational persistence."""

    def __init__(
        self,
        graph_engine: DependencyGraphEngine,
        metadata_store: Optional[OperationalMetadataStore] = None,
        max_alerts: int = 150,
    ):
        self.graph = graph_engine
        self.metadata_store = metadata_store or OperationalMetadataStore()
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_budget_max_firing = 5
        self.suppressed_symptom_count = 0
        self.suppressed_budget_count = 0

    def process_anomaly(self, anomaly: AnomalyRecord, active_origins: Optional[List[str]] = None) -> Dict[str, Any]:
        origins = active_origins or []
        is_downstream_symptom = False

        # Downstream symptom suppression based on directed dependency traversal
        for origin in origins:
            if origin != anomaly.service_id and origin in self.graph.get_upstream_dependencies(anomaly.service_id):
                is_downstream_symptom = True
                break

        # Check alert budget
        firing_count = sum(
            1 for a in self.alerts
            if (a.get("status") in [AlertStatus.FIRING.value, "FIRING"])
        )
        budget_exceeded = firing_count >= self.alert_budget_max_firing

        status = AlertStatus.FIRING
        reason = "Direct anomaly detected"
        if is_downstream_symptom:
            status = AlertStatus.SUPPRESSED
            reason = f"Suppressed as downstream symptom of upstream candidate {', '.join(origins)}"
            self.suppressed_symptom_count += 1
        elif budget_exceeded:
            status = AlertStatus.SUPPRESSED
            reason = "Suppressed to preserve operational alert budget quota (max 5 simultaneous firing alerts)"
            self.suppressed_budget_count += 1

        alert_entry = {
            "alert_id": f"alert-{uuid.uuid4().hex[:8]}",
            "service_id": anomaly.service_id,
            "anomaly_id": anomaly.anomaly_id,
            "status": status.value,
            "severity": anomaly.severity.value if hasattr(anomaly.severity, "value") else str(anomaly.severity),
            "suppression_reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": round(anomaly.score, 3),
            "acknowledged_by": None,
        }
        self.alerts.append(alert_entry)

        # Persist to relational metadata store
        if self.metadata_store:
            self.metadata_store.save_alert(alert_entry)

        return alert_entry

    def acknowledge_alert(self, alert_id: str, user: str = "oncall-analyst") -> Optional[Dict[str, Any]]:
        for a in self.alerts:
            if a["alert_id"] == alert_id:
                a["status"] = AlertStatus.ACKNOWLEDGED.value
                a["acknowledged_by"] = user
                if self.metadata_store:
                    self.metadata_store.acknowledge_alert(alert_id, user)
                return a
        if self.metadata_store:
            if self.metadata_store.acknowledge_alert(alert_id, user):
                return {"alert_id": alert_id, "status": "ACKNOWLEDGED", "acknowledged_by": user}
        return None

    def get_all_alerts(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if self.metadata_store:
            db_alerts = self.metadata_store.list_alerts(status=status, limit=limit)
            if db_alerts:
                return db_alerts
        results = list(self.alerts)
        if status:
            results = [a for a in results if a.get("status", "").upper() == status.upper()]
        return results[-limit:]

    def get_budget_status(self) -> Dict[str, Any]:
        firing = sum(1 for a in self.alerts if a.get("status") in ["FIRING", AlertStatus.FIRING.value])
        return {
            "max_firing_budget": self.alert_budget_max_firing,
            "current_firing_count": firing,
            "available_alert_quota": max(0, self.alert_budget_max_firing - firing),
            "total_suppressed_as_symptoms": self.suppressed_symptom_count,
            "total_suppressed_for_budget": self.suppressed_budget_count,
        }
