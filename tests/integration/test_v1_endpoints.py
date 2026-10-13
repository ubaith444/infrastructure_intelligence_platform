from datetime import datetime, timezone
from fastapi.testclient import TestClient
import pytest
from apps.api.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_v1_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "dependencies" in data
    assert "broker" in data["dependencies"]
    assert "lakehouse" in data["dependencies"]
    assert "anomaly_arbiter" in data["dependencies"]
    assert "risk_predictor" in data["dependencies"]
    assert "capacity_forecaster" in data["dependencies"]
    assert "dependency_graph" in data["dependencies"]

def test_v1_services_inventory(client):
    res = client.get("/api/v1/services")
    assert res.status_code == 200
    data = res.json()
    assert "services" in data
    assert len(data["services"]) > 0
    svc = data["services"][0]
    assert "id" in svc
    assert "tier" in svc
    assert "status" in svc
    assert "current_metrics" in svc
    assert "incident_risk_probability" in svc
    assert "dependencies" in svc
    assert "callers" in svc

def test_v1_service_metrics(client):
    res = client.get("/api/v1/services/payment-service/metrics?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["service_id"] == "payment-service"
    assert "metrics" in data
    assert len(data["metrics"]) > 0
    pt = data["metrics"][0]
    assert "timestamp" in pt
    assert "latency_p50" in pt
    assert "error_rate" in pt
    assert "cpu_utilization" in pt

def test_v1_service_anomalies(client):
    res = client.get("/api/v1/services/payment-service/anomalies?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["service_id"] == "payment-service"
    assert "anomalies" in data

def test_v1_incidents_list(client):
    res = client.get("/api/v1/incidents?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "incidents" in data
    assert len(data["incidents"]) > 0
    inc = data["incidents"][0]
    assert "incident_id" in inc
    assert "title" in inc
    assert "status" in inc

def test_v1_incident_detail(client):
    res_list = client.get("/api/v1/incidents?limit=5")
    inc_id = res_list.json()["incidents"][0]["incident_id"]

    res = client.get(f"/api/v1/incidents/{inc_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["incident_id"] == inc_id
    assert "timeline" in data
    assert "candidate_rankings" in data

def test_v1_dependencies_graph(client):
    res = client.get("/api/v1/dependencies")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "links" in data
    assert data["total_nodes"] > 0
    assert data["total_edges"] > 0

def test_v1_forecasts(client):
    res = client.get("/api/v1/forecasts/checkout-api?metric=request_rate&capacity_limit=300")
    assert res.status_code == 200
    data = res.json()
    assert data["service_id"] == "checkout-api"
    assert "forecasts" in data
    assert "horizon_5m" in data["forecasts"]
    assert "horizon_15m" in data["forecasts"]
    assert "horizon_60m" in data["forecasts"]
    h5 = data["forecasts"]["horizon_5m"]
    assert "predicted_median" in h5
    assert "interval_p10" in h5
    assert "interval_p90" in h5

def test_v1_pipeline_status(client):
    res = client.get("/api/v1/pipeline/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "throughput_eps" in data
    assert "consumer_lag" in data
    assert "dead_letter_queue_size" in data
    assert "lakehouse_records" in data

def test_v1_models_evaluation(client):
    res = client.get("/api/v1/models/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "robust_statistical_mad" in data["models"]
    assert "multivariate_isolation_forest" in data["models"]
    assert "river_streaming_halfspace" in data["models"]
    assert "incident_risk_predictor" in data["models"]
    assert "workload_capacity_forecaster" in data["models"]
    assert "feature_drift_monitoring" in data

def test_v1_fault_simulation_lifecycle(client):
    # Inject fault
    res_inject = client.post("/api/v1/simulations/faults", json={"scenario": "payment_cascade", "duration_seconds": 60})
    assert res_inject.status_code == 200
    data_inj = res_inject.json()
    assert data_inj["status"] == "FAULT_TRIGGERED"
    assert data_inj["scenario"] == "payment_cascade"
    assert data_inj["affected_service"] == "payment-service"

    # Check that incident list reflects the fault
    res_inc = client.get("/api/v1/incidents")
    assert res_inc.status_code == 200
    assert any("payment" in inc["incident_id"] for inc in res_inc.json()["incidents"])

    # Clear fault
    res_clear = client.post("/api/v1/simulations/faults", json={"scenario": "clear"})
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "FAULT_CLEARED"

def test_v1_telemetry_ingestion(client):
    now = datetime.now(timezone.utc)
    evt = {
        "event_id": "test-ingest-v1-01",
        "event_time": now.isoformat(),
        "service": "order-service",
        "environment": "production",
        "event_type": "service_metric",
        "metric_name": "request_latency_ms",
        "metric_value": 42.5,
        "unit": "ms"
    }
    res = client.post("/api/v1/telemetry/ingest", json={"events": [evt]})
    assert res.status_code == 200
    data = res.json()
    assert data["ingested"] == 1
    assert data["rejected"] == 0
