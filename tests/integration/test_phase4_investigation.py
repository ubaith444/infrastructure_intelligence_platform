from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from apps.api.server import app
from storage.metadata_store import OperationalMetadataStore
from storage.models import AlertSeverity, AlertStatus, AnomalyRecord, TelemetryEvent


@pytest.fixture
def client():
    return TestClient(app)


def test_telemetry_event_cardinality_and_schema_validation():
    now = datetime.now(timezone.utc)
    # 1. Valid event
    evt_valid = TelemetryEvent(
        event_id="evt-000001",
        event_time=now,
        ingested_at=now,
        service="checkout-api",
        environment="development",
        event_type="service_metric",
        metric_name="request_latency_ms",
        metric_value=425.7,
        unit="ms",
        host_id="host-03",
        schema_version="1.0",
        attributes={"endpoint": "/pay", "status_code": "200"},
    )
    assert evt_valid.event_id == "evt-000001"
    assert evt_valid.schema_version == "1.0"

    # 2. Reject unbounded cardinality labels (banned keys: user_id, raw_url)
    with pytest.raises(ValueError, match="Banned unbounded attribute label"):
        TelemetryEvent(
            event_id="evt-000002",
            event_time=now,
            service="checkout-api",
            metric_name="latency",
            metric_value=120.0,
            attributes={"user_id": "usr-9992314"},
        )

    # 3. Reject excess attribute tags (>25)
    too_many_tags = {f"tag_{i}": f"val_{i}" for i in range(30)}
    with pytest.raises(ValueError, match="cardinality exceeded limit"):
        TelemetryEvent(
            event_id="evt-000003",
            event_time=now,
            service="checkout-api",
            metric_name="latency",
            metric_value=120.0,
            attributes=too_many_tags,
        )


def test_operational_metadata_store_core_entities(tmp_path):
    store = OperationalMetadataStore(db_path=tmp_path / "metadata.db")

    # Services and dependencies seeded
    services = store.list_services()
    assert len(services) >= 5
    deps = store.list_dependencies()
    assert len(deps) >= 5

    # Save incident and evidence
    now_str = datetime.now(timezone.utc).isoformat()
    store.save_incident({
        "incident_id": "inc-test-01",
        "title": "Database CPU Saturation",
        "severity": "CRITICAL",
        "status": "INVESTIGATING",
        "start_time": now_str,
        "service_count": 2,
        "primary_candidate_origin": "db-primary",
        "origin_confidence": 0.92,
        "candidate_rankings": [{"service_id": "db-primary", "confidence": 0.92}],
        "timeline": [{"timestamp": now_str, "event_type": "ANOMALY", "service": "db-primary", "message": "CPU 95%"}],
    })

    inc = store.get_incident("inc-test-01")
    assert inc is not None
    assert inc["primary_candidate_origin"] == "db-primary"
    assert len(inc["timeline"]) == 1

    # Save evidence
    store.save_incident_evidence(
        incident_id="inc-test-01",
        related_anomalies=["anom-db-01"],
        contributing_signals=[{"metric": "cpu", "value": 95.0}],
        associative_evidence="High query queue depth correlated with lock contention.",
    )
    evidence = store.get_incident_evidence("inc-test-01")
    assert len(evidence) == 1
    assert "anom-db-01" in evidence[0]["related_anomalies"]


def test_v1_alerts_and_acknowledgment(client):
    res = client.get("/api/v1/alerts")
    assert res.status_code == 200
    data = res.json()
    assert "total_alerts" in data
    assert "alerts" in data
    assert "alert_budget" in data
    assert "max_firing_budget" in data["alert_budget"]

    # Trigger or acknowledge an alert
    from apps.api.routes import alert_manager
    anom = AnomalyRecord(
        anomaly_id="anom-test-alert",
        service_id="payment-service",
        detector_type="RobustStatisticalMAD",
        timestamp=datetime.now(timezone.utc),
        severity=AlertSeverity.CRITICAL,
        score=5.5,
        threshold=3.5,
        narrative="Payment error spike",
    )
    alert = alert_manager.process_anomaly(anom)
    assert alert["status"] in [AlertStatus.FIRING.value, AlertStatus.SUPPRESSED.value]

    # Test acknowledgment endpoint
    res_ack = client.post(f"/api/v1/alerts/{alert['alert_id']}/acknowledge?user=secops-lead")
    assert res_ack.status_code == 200
    assert res_ack.json()["status"] == "ACKNOWLEDGED"


def test_v1_incident_evidence_and_feedback(client):
    # Fetch existing baseline incident detail
    res_inc = client.get("/api/v1/incidents/inc-baseline-01")
    assert res_inc.status_code == 200
    inc_data = res_inc.json()
    assert inc_data["incident_id"] == "inc-baseline-01"
    assert "timeline" in inc_data

    # Fetch incident evidence
    res_ev = client.get("/api/v1/incidents/inc-baseline-01/evidence")
    assert res_ev.status_code == 200
    ev_data = res_ev.json()
    assert ev_data["incident_id"] == "inc-baseline-01"
    assert "evidence" in ev_data
    assert len(ev_data["evidence"]) > 0

    # Submit analyst feedback
    feedback_payload = {
        "incident_id": "inc-baseline-01",
        "suggested_origin": "payment-service",
        "verdict": "CONFIRMED",
        "actual_origin": "payment-service",
    }
    res_fb = client.post("/api/v1/incidents/inc-baseline-01/feedback", json=feedback_payload)
    assert res_fb.status_code == 200
    fb_data = res_fb.json()
    assert fb_data["status"] == "FEEDBACK_RECORDED"
    assert fb_data["verdict"] == "CONFIRMED"
    assert "calibrated_damping" in fb_data


def test_v1_metadata_services_and_runs(client):
    res_services = client.get("/api/v1/metadata/services")
    assert res_services.status_code == 200
    data_svc = res_services.json()
    assert "services" in data_svc
    assert "dependencies" in data_svc
    assert data_svc["total_services"] >= 5

    res_runs = client.get("/api/v1/metadata/runs")
    assert res_runs.status_code == 200
    data_runs = res_runs.json()
    assert "runs" in data_runs
