from fastapi.testclient import TestClient
import pytest
from apps.api.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_pipeline_health_endpoint(client):
    res = client.get("/api/pipeline/health")
    assert res.status_code == 200
    data = res.json()
    assert "health" in data

def test_services_health_endpoint(client):
    res = client.get("/api/services/health")
    assert res.status_code == 200
    data = res.json()
    assert "services" in data

def test_security_headers_and_status(client):
    res = client.get("/api/security/status")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    data = res.json()
    assert "auth_enabled" in data
    assert "client_rate_limit" in data
