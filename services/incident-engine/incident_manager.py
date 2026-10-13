from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from services.incident_engine.candidate_ranker import CandidateOriginRanker
from services.incident_engine.graph import DependencyGraphEngine
from storage.metadata_store import OperationalMetadataStore
from storage.models import AlertSeverity, AnomalyRecord, IncidentRecord, IncidentStatus


class IncidentManager:
    """Manages full lifecycle of operational incidents, associative evidence, and timeline reconstruction."""

    def __init__(
        self,
        graph_engine: DependencyGraphEngine,
        ranker: CandidateOriginRanker,
        metadata_store: Optional[OperationalMetadataStore] = None,
    ):
        self.graph = graph_engine
        self.ranker = ranker
        self.metadata_store = metadata_store or OperationalMetadataStore()
        self.incidents: Dict[str, Dict[str, Any]] = {}
        self._history: deque = deque(maxlen=50)
        self.active_incident_id: Optional[str] = None
        self._seed_sample_incidents()

    def _seed_sample_incidents(self) -> None:
        """Seeds initial historical baseline incidents for auditability."""
        now = datetime.now(timezone.utc)
        sample_id = "inc-baseline-01"
        self.incidents[sample_id] = {
            "incident_id": sample_id,
            "title": "Payment Authorization Latency Cascade",
            "status": "RESOLVED",
            "severity": "CRITICAL",
            "started_at": datetime.fromtimestamp(now.timestamp() - 7200, tz=timezone.utc).isoformat(),
            "updated_at": datetime.fromtimestamp(now.timestamp() - 3600, tz=timezone.utc).isoformat(),
            "resolved_at": datetime.fromtimestamp(now.timestamp() - 3600, tz=timezone.utc).isoformat(),
            "service_count": 3,
            "services_involved": ["payment-service", "checkout-api", "edge-gateway"],
            "primary_candidate_origin": "payment-service",
            "origin_confidence": 0.885,
            "candidate_rankings": [
                {
                    "service_id": "payment-service",
                    "confidence": 0.885,
                    "temporal_score": 0.98,
                    "topology_score": 0.90,
                    "severity_score": 0.95,
                    "associative_narrative": "Service 'payment-service' initiated the earliest anomaly cluster with critical latency degradation.",
                    "blast_radius": ["checkout-api", "edge-gateway", "payment-service"],
                },
                {
                    "service_id": "checkout-api",
                    "confidence": 0.542,
                    "temporal_score": 0.45,
                    "topology_score": 0.60,
                    "severity_score": 0.60,
                    "associative_narrative": "Downstream symptom caller of payment-service.",
                    "blast_radius": ["checkout-api", "edge-gateway"],
                },
            ],
            "timeline": [
                {
                    "timestamp": datetime.fromtimestamp(now.timestamp() - 7200, tz=timezone.utc).isoformat(),
                    "event_type": "ANOMALY_TRIGGERED",
                    "service": "payment-service",
                    "message": "Payment latency p90 surged to 280ms (RobustStatisticalMAD z=6.4)",
                    "severity": "CRITICAL",
                },
                {
                    "timestamp": datetime.fromtimestamp(now.timestamp() - 7140, tz=timezone.utc).isoformat(),
                    "event_type": "CASCADE_DETECTED",
                    "service": "checkout-api",
                    "message": "Checkout API thread pool saturation observed; latency rose to 195ms",
                    "severity": "WARNING",
                },
                {
                    "timestamp": datetime.fromtimestamp(now.timestamp() - 3600, tz=timezone.utc).isoformat(),
                    "event_type": "RESOLVED",
                    "service": "payment-service",
                    "message": "Downstream payment gateway reconnected; circuit breaker reset",
                    "severity": "INFO",
                },
            ],
            "blast_radius": ["payment-service", "checkout-api", "edge-gateway"],
            "evidence": [
                {
                    "detector": "RobustStatisticalMAD",
                    "metric": "request_latency_ms",
                    "observed_value": 280.0,
                    "baseline": 45.0,
                    "deviation_sigma": 6.4,
                },
                {
                    "detector": "MultivariateIsolationForest",
                    "metric": "error_rate",
                    "observed_value": 0.18,
                    "baseline": 0.002,
                    "subspace_contribution": 0.85,
                },
            ],
        }

        # Persist baseline incident to metadata store
        if self.metadata_store:
            self.metadata_store.save_incident(self.incidents[sample_id])
            self.metadata_store.save_incident_evidence(
                incident_id=sample_id,
                related_anomalies=["anom-payment-01", "anom-checkout-02"],
                contributing_signals=self.incidents[sample_id]["evidence"],
                associative_evidence="Downstream checkout latency rose following payment failure; upstream dependency correlation.",
            )

    def sync_active_fault(self, fault_status: Dict[str, Any], recent_anomalies: List[AnomalyRecord]) -> Optional[Dict[str, Any]]:
        """Syncs the incident state with active fault injection and recent anomalies."""
        now_iso = datetime.now(timezone.utc).isoformat()
        if fault_status and fault_status.get("active"):
            scenario = fault_status.get("scenario", "unknown_fault")
            service = fault_status.get("service", "unknown_service")
            inc_id = f"inc-{scenario}"

            if inc_id not in self.incidents:
                # Create incident
                blast = self.graph.get_blast_radius(service)
                candidates = self.ranker.rank_candidates(recent_anomalies)
                self.incidents[inc_id] = {
                    "incident_id": inc_id,
                    "title": f"Fault Injection: {fault_status.get('description', scenario)}",
                    "status": "INVESTIGATING",
                    "severity": "CRITICAL",
                    "started_at": fault_status.get("injected_at", now_iso),
                    "updated_at": now_iso,
                    "resolved_at": None,
                    "service_count": len(blast),
                    "services_involved": blast,
                    "primary_candidate_origin": service,
                    "origin_confidence": 0.95,
                    "candidate_rankings": candidates,
                    "timeline": [
                        {
                            "timestamp": fault_status.get("injected_at", now_iso),
                            "event_type": "FAULT_INJECTED",
                            "service": service,
                            "message": f"Controlled fault scenario '{scenario}' injected on {service}",
                            "severity": "CRITICAL",
                        }
                    ],
                    "blast_radius": blast,
                    "evidence": [],
                }
                self.active_incident_id = inc_id
            else:
                inc = self.incidents[inc_id]
                inc["updated_at"] = now_iso
                inc["status"] = "INVESTIGATING"
                inc["candidate_rankings"] = self.ranker.rank_candidates(recent_anomalies)

                # Add new timeline events from anomalies if not recorded
                for a in recent_anomalies[-5:]:
                    if a.service_id in inc["services_involved"]:
                        time_str = a.timestamp.isoformat()
                        if not any(t.get("timestamp") == time_str for t in inc["timeline"]):
                            inc["timeline"].append({
                                "timestamp": time_str,
                                "event_type": "ANOMALY_RECORDED",
                                "service": a.service_id,
                                "message": a.narrative,
                                "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                            })
                            inc["evidence"].append({
                                "detector": a.detector_type,
                                "service": a.service_id,
                                "score": a.score,
                                "attributions": a.metric_attributions,
                            })

            # Sync to metadata store
            if self.metadata_store:
                self.metadata_store.save_incident(self.incidents[inc_id])
                if self.incidents[inc_id]["evidence"]:
                    self.metadata_store.save_incident_evidence(
                        incident_id=inc_id,
                        related_anomalies=[a.anomaly_id for a in recent_anomalies if a.service_id in self.incidents[inc_id]["services_involved"]],
                        contributing_signals=self.incidents[inc_id]["evidence"][-3:],
                        associative_evidence=f"Active cascade associated with {service} fault scenario.",
                    )

            return self.incidents[inc_id]
        else:
            # If an active incident was running and fault is cleared, resolve it
            if self.active_incident_id and self.active_incident_id in self.incidents:
                inc = self.incidents[self.active_incident_id]
                if inc["status"] != "RESOLVED":
                    inc["status"] = "RESOLVED"
                    inc["resolved_at"] = now_iso
                    inc["updated_at"] = now_iso
                    inc["timeline"].append({
                        "timestamp": now_iso,
                        "event_type": "RESOLVED",
                        "service": inc["primary_candidate_origin"],
                        "message": "Fault cleared. Metrics recovered to rolling baselines.",
                        "severity": "INFO",
                    })
                    if self.metadata_store:
                        self.metadata_store.save_incident(inc)
                self.active_incident_id = None
        return None

    def list_incidents(
        self,
        status: Optional[str] = None,
        service_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        results = list(self.incidents.values())
        if status:
            results = [i for i in results if i.get("status", "").upper() == status.upper()]
        if service_id:
            results = [i for i in results if service_id in i.get("services_involved", []) or i.get("primary_candidate_origin") == service_id]
        if severity:
            results = [i for i in results if i.get("severity", "").upper() == severity.upper()]

        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        return self.incidents.get(incident_id)

    def get_incident_evidence(self, incident_id: str) -> List[Dict[str, Any]]:
        inc = self.get_incident(incident_id)
        if not inc:
            return []
        stored_evidence = self.metadata_store.get_incident_evidence(incident_id) if self.metadata_store else []
        if stored_evidence:
            return stored_evidence
        return inc.get("evidence", [])

    def record_feedback(
        self,
        incident_id: str,
        suggested_origin: str,
        verdict: str,
        actual_origin: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records analyst feedback, updates incident origin confidence, and persists to metadata store."""
        inc = self.get_incident(incident_id)
        if inc:
            if verdict == "CONFIRMED":
                inc["origin_confidence"] = min(0.99, inc.get("origin_confidence", 0.8) + 0.05)
            elif verdict == "INCORRECT_ORIGIN":
                inc["primary_candidate_origin"] = actual_origin or suggested_origin
                inc["origin_confidence"] = 0.50
            elif verdict == "FALSE_POSITIVE":
                inc["status"] = "FALSE_POSITIVE"

            if self.metadata_store:
                self.metadata_store.save_incident(inc)

        if self.metadata_store:
            return self.metadata_store.save_analyst_feedback(
                incident_id=incident_id,
                suggested_origin=suggested_origin,
                verdict=verdict,
                actual_origin=actual_origin,
            )
        return {
            "incident_id": incident_id,
            "suggested_origin": suggested_origin,
            "verdict": verdict,
            "actual_origin": actual_origin or suggested_origin,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

