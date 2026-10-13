import pytest
from fastapi.testclient import TestClient

from apps.api.routes import sec_manager
from apps.api.server import app
from storage.metadata_store import OperationalMetadataStore


@pytest.fixture
def client():
    return TestClient(app)


def test_v1_benchmark_report_endpoint(client):
    res = client.get("/api/v1/benchmarks/report")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark_id" in data
    assert "environment" in data
    assert "ingestion_performance" in data
    assert "anomaly_detection_smd_benchmark" in data
    assert "incident_risk_model_evaluation" in data
    assert "capacity_forecaster_evaluation" in data
    assert "distributed_processing_decision_matrix" in data

    # Check detector benchmarks
    detectors = data["anomaly_detection_smd_benchmark"]["detectors"]
    assert "robust_statistical_mad" in detectors
    assert "multivariate_isolation_forest" in detectors
    assert "river_streaming_halfspace" in detectors
    assert detectors["robust_statistical_mad"]["point_adjusted_f1"] > 0.8

    # Check distributed decision matrix
    dist_matrix = data["distributed_processing_decision_matrix"]
    assert dist_matrix["recommendation"] == "RETAIN_SINGLE_NODE_ARCHITECTURE"
    assert "distributed_migration_triggers" in dist_matrix
    assert len(dist_matrix["distributed_migration_triggers"]) >= 3


def test_v1_benchmark_on_demand_run(client):
    res = client.post("/api/v1/benchmarks/run?n_smd_points=150&n_ingest_events=200")
    assert res.status_code == 200
    data = res.json()
    assert data["overall_status"] == "PASS"
    assert data["ingestion_performance"]["events_benchmarked"] == 200


def test_v1_models_drift_endpoint(client):
    res = client.get("/api/v1/models/drift")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "max_psi" in data
    assert "feature_drift" in data
    assert "latency_percentiles" in data
    assert "retrain_recommended" in data


def test_v1_models_retrain_endpoint(client):
    res = client.post("/api/v1/models/retrain")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "MODEL_RETRAINED"
    assert "run_id" in data
    assert "new_version" in data
    assert data["drift_baseline_reset"] is True

    # Verify run was persisted in metadata store
    store = OperationalMetadataStore()
    runs = store.list_model_runs(limit=5)
    assert len(runs) > 0
    assert any(r["run_id"] == data["run_id"] for r in runs)


def test_v1_models_retrain_rbac_enforcement(client):
    orig_auth = sec_manager.auth_enabled
    try:
        sec_manager.auth_enabled = True

        # Viewer role key should be forbidden (403)
        res_viewer = client.post("/api/v1/models/retrain", headers={"X-API-Key": "iip-viewer-key"})
        assert res_viewer.status_code == 403

        # Operator role key should succeed (200)
        res_operator = client.post("/api/v1/models/retrain", headers={"X-API-Key": "iip-demo-key-2026"})
        assert res_operator.status_code == 200

        # Admin role key should succeed (200)
        res_admin = client.post("/api/v1/models/retrain", headers={"X-API-Key": "iip-key-prod-01"})
        assert res_admin.status_code == 200

    finally:
        sec_manager.auth_enabled = orig_auth


def test_fault_simulation_input_sanitization(client):
    # Valid scenario
    res_valid = client.post("/api/v1/simulations/faults", json={"scenario": "clear"})
    assert res_valid.status_code == 200

    # Malicious injection attempt with script tags
    with pytest.raises(ValueError, match="forbidden HTML/Script markup tags"):
        client.post("/api/v1/simulations/faults", json={"scenario": "<script>alert(1)</script>"})

    # Malicious injection attempt with SQL syntax
    with pytest.raises(ValueError, match="forbidden SQL syntax patterns"):
        client.post("/api/v1/simulations/faults", json={"scenario": "payment_cascade'; DROP TABLE services; --"})
