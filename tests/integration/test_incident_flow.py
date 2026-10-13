from datetime import datetime, timezone
import pytest
from services.incident_engine.alert_manager import AlertManager
from services.incident_engine.candidate_ranker import CandidateOriginRanker
from services.incident_engine.feedback import AnalystFeedbackEngine
from services.incident_engine.graph import DependencyGraphEngine
from storage.models import AlertSeverity, AlertStatus, AnomalyRecord

def test_incident_candidate_ranking_and_alert_suppression():
    graph = DependencyGraphEngine()
    ranker = CandidateOriginRanker(graph)
    alert_mgr = AlertManager(graph)

    now = datetime.now(timezone.utc)
    # Payment fails first, checkout fails as caller
    a_payment = AnomalyRecord(
        anomaly_id="a1",
        service_id="payment-service",
        detector_type="MAD",
        timestamp=now,
        severity=AlertSeverity.CRITICAL,
        score=6.0,
        narrative="Payment error surge"
    )
    a_checkout = AnomalyRecord(
        anomaly_id="a2",
        service_id="checkout-api",
        detector_type="MAD",
        timestamp=now,
        severity=AlertSeverity.WARNING,
        score=3.8,
        narrative="Checkout latency rise"
    )

    candidates = ranker.rank_candidates([a_payment, a_checkout])
    assert len(candidates) == 2
    # payment-service is upstream dependency of checkout-api and has higher severity -> higher confidence
    assert candidates[0]["service_id"] == "payment-service"

    # Alert manager should suppress checkout-api alert as symptom of payment-service
    alert1 = alert_mgr.process_anomaly(a_payment, active_origins=["payment-service"])
    alert2 = alert_mgr.process_anomaly(a_checkout, active_origins=["payment-service"])

    assert alert1["status"] == AlertStatus.FIRING.value
    assert alert2["status"] == AlertStatus.SUPPRESSED.value
    assert "downstream symptom" in alert2["suppression_reason"].lower()
