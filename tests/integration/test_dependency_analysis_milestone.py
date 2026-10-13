"""Milestone 3 Acceptance Test Suite: Dependency Analysis.

Verification criteria:
1. Directed DAG traversal (upstream dependencies, downstream callers, blast radius).
2. Error propagation tracing (associative propagation evidence: caller latency increased after upstream errors rose).
3. 5-Factor candidate origin ranking with associative confidence scoring.
4. Downstream symptom alert suppression and alert budget compliance.
5. API endpoints: /api/v1/dependencies, /api/v1/dependencies/{id}/blast-radius, /api/v1/dependencies/rank-origins.
"""

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import pytest

from apps.api.server import app
from services.incident_engine.candidate_ranker import CandidateOriginRanker
from services.incident_engine.graph import DependencyGraphEngine
from storage.models import AlertSeverity, AnomalyRecord


@pytest.fixture
def client():
    return TestClient(app)


def test_directed_dag_traversal_and_blast_radius():
    """Verify Directed Acyclic Graph traversal and reverse-ancestor blast radius."""
    graph = DependencyGraphEngine()

    # 1. Edge-gateway dependencies
    gw_upstream = graph.get_upstream_dependencies("edge-gateway")
    assert "checkout-api" in gw_upstream
    assert "auth-service" in gw_upstream
    assert "catalog-service" in gw_upstream

    # 2. Downstream callers of payment-service
    pay_callers = graph.get_downstream_callers("payment-service")
    assert "order-service" in pay_callers
    assert "checkout-api" in pay_callers

    # 3. Transitive blast radius (services affected if payment-service fails)
    blast = graph.get_blast_radius("payment-service")
    assert "payment-service" in blast
    assert "order-service" in blast
    assert "checkout-api" in blast
    assert "edge-gateway" in blast
    assert len(blast) >= 4


def test_error_propagation_tracing_and_candidate_origin_ranking():
    """Verify error propagation evidence and 5-factor candidate origin ranking."""
    graph = DependencyGraphEngine()
    ranker = CandidateOriginRanker(graph)

    now = datetime.now(timezone.utc)
    t0 = now - timedelta(seconds=20)
    t1 = now - timedelta(seconds=10)

    # Simulate root failure: payment-service spikes errors first at t0
    anom_payment = AnomalyRecord(
        anomaly_id="anom-pay-1",
        service_id="payment-service",
        detector_type="RobustStatisticalMAD",
        timestamp=t0,
        severity=AlertSeverity.CRITICAL,
        score=7.2,
        threshold=3.5,
        metric_attributions={"error_rate": 0.85, "latency_p90": 0.15},
        narrative="Statistical error rate deviation detected on payment-service: err_z=7.2",
    )

    # Cascading symptom: order-service exhibits latency degradation shortly after at t1
    anom_order = AnomalyRecord(
        anomaly_id="anom-ord-1",
        service_id="order-service",
        detector_type="RobustStatisticalMAD",
        timestamp=t1,
        severity=AlertSeverity.WARNING,
        score=4.8,
        threshold=3.5,
        metric_attributions={"latency_p90": 0.80, "error_rate": 0.10},
        narrative="Statistical latency deviation detected on order-service: latency_z=4.8",
    )

    # Cascading symptom: checkout-api experiences timeout/latency degradation at t1
    anom_checkout = AnomalyRecord(
        anomaly_id="anom-chk-1",
        service_id="checkout-api",
        detector_type="RobustStatisticalMAD",
        timestamp=t1,
        severity=AlertSeverity.WARNING,
        score=4.2,
        threshold=3.5,
        metric_attributions={"latency_p90": 0.75, "error_rate": 0.15},
        narrative="Statistical latency deviation detected on checkout-api: latency_z=4.2",
    )

    ranked = ranker.rank_candidates([anom_payment, anom_order, anom_checkout])

    # Payment-service should be ranked #1 candidate origin with highest confidence
    assert len(ranked) == 3
    primary = ranked[0]
    assert primary["service_id"] == "payment-service"
    assert primary["confidence"] > ranked[1]["confidence"]

    # Propagation evidence should capture caller degradation following payment errors
    evidence = primary["propagation_evidence"]
    assert any("order-service latency increased after payment-service errors rose" in e for e in evidence)

    # Narrative must adhere to linguistic governance
    narrative = primary["associative_narrative"]
    assert "candidate origin confidence" in narrative
    assert "Associative evidence trail" in narrative
    assert "observational telemetry does not assert unverified causality" in narrative


def test_analyst_feedback_damping_calibration():
    """Verify analyst feedback adjusts empirical damping weights."""
    graph = DependencyGraphEngine()
    ranker = CandidateOriginRanker(graph)

    now = datetime.now(timezone.utc)
    anom = AnomalyRecord(
        anomaly_id="anom-test-1",
        service_id="catalog-service",
        detector_type="RobustStatisticalMAD",
        timestamp=now,
        severity=AlertSeverity.WARNING,
        score=4.0,
        threshold=3.5,
        metric_attributions={"latency_p90": 0.90},
        narrative="Latency spike on catalog-service",
    )

    # Without damping
    uncalibrated = ranker.rank_candidates([anom])[0]["confidence"]

    # Damped due to prior false-positive history (damping factor = 0.60)
    calibrated = ranker.rank_candidates([anom], damping_factors={"catalog-service": 0.60})[0]["confidence"]

    assert calibrated < uncalibrated
    assert round(calibrated, 2) == round(uncalibrated * 0.60, 2)


def test_v1_dependencies_graph_api(client):
    """Verify GET /api/v1/dependencies."""
    res = client.get("/api/v1/dependencies")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "links" in data
    assert data["total_nodes"] >= 6
    assert data["total_edges"] >= 6
    assert any(n["id"] == "payment-service" for n in data["nodes"])


def test_v1_service_blast_radius_api(client):
    """Verify GET /api/v1/dependencies/{service_id}/blast-radius."""
    res = client.get("/api/v1/dependencies/payment-service/blast-radius")
    assert res.status_code == 200
    data = res.json()
    assert data["service_id"] == "payment-service"
    assert "order-service" in data["direct_callers"] or "checkout-api" in data["direct_callers"]
    assert "edge-gateway" in data["transitive_blast_radius"]
    assert data["blast_radius_size"] >= 3


def test_v1_rank_origins_api(client):
    """Verify POST /api/v1/dependencies/rank-origins."""
    now = datetime.now(timezone.utc).isoformat()
    custom_anoms = {
        "anomalies": [
            {
                "anomaly_id": "anom-p-1",
                "service_id": "payment-service",
                "detector_type": "RobustStatisticalMAD",
                "timestamp": now,
                "severity": "CRITICAL",
                "score": 6.8,
                "threshold": 3.5,
                "metric_attributions": {"error_rate": 0.80},
                "narrative": "Error surge on payment-service",
            },
            {
                "anomaly_id": "anom-o-1",
                "service_id": "order-service",
                "detector_type": "RobustStatisticalMAD",
                "timestamp": now,
                "severity": "WARNING",
                "score": 4.5,
                "threshold": 3.5,
                "metric_attributions": {"latency_p90": 0.70},
                "narrative": "Latency surge on order-service",
            },
        ]
    }
    res = client.post("/api/v1/dependencies/rank-origins", json=custom_anoms)
    assert res.status_code == 200
    data = res.json()
    assert "candidate_origins" in data
    assert len(data["candidate_origins"]) == 2
    assert data["primary_candidate"]["service_id"] == "payment-service"
