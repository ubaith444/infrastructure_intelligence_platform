import pytest
from apps.api.security import SecurityManager, SlidingWindowRateLimiter, UserRole
from apps.api.server import app
from fastapi.testclient import TestClient


def test_api_key_and_rbac():
    sm = SecurityManager()
    sm.auth_enabled = True

    # Valid keys
    assert sm.validate_api_key("iip-key-prod-01") is True
    assert sm.validate_api_key("iip-demo-key-2026") is True
    assert sm.validate_api_key("iip-analyst-key") is True
    assert sm.validate_api_key("iip-viewer-key") is True

    # Invalid keys
    assert sm.validate_api_key("invalid-token-123") is False
    assert sm.validate_api_key("") is False
    assert sm.validate_api_key(None) is False

    # RBAC permissions
    assert sm.check_permission("iip-key-prod-01", UserRole.ADMIN) is True
    assert sm.check_permission("iip-key-prod-01", UserRole.OPERATOR) is True
    assert sm.check_permission("iip-key-prod-01", UserRole.VIEWER) is True

    assert sm.check_permission("iip-demo-key-2026", UserRole.OPERATOR) is True
    assert sm.check_permission("iip-demo-key-2026", UserRole.ADMIN) is False

    assert sm.check_permission("iip-viewer-key", UserRole.VIEWER) is True
    assert sm.check_permission("iip-viewer-key", UserRole.OPERATOR) is False
    assert sm.check_permission("iip-viewer-key", UserRole.ADMIN) is False


def test_rate_limiter_burst_and_sustained():
    limiter = SlidingWindowRateLimiter(requests_per_minute=20, burst_limit=5)
    client_id = "test-client-ip"

    # First 5 calls pass burst limit
    for _ in range(5):
        allowed, _ = limiter.is_allowed(client_id)
        assert allowed is True

    # 6th immediate call hits burst limit
    allowed, retry_after = limiter.is_allowed(client_id)
    assert allowed is False
    assert retry_after == 1

    headers = limiter.get_rate_limit_headers(client_id)
    assert "X-RateLimit-Limit" in headers
    assert "X-RateLimit-Remaining" in headers
    assert int(headers["X-RateLimit-Limit"]) == 20


def test_input_sanitization():
    sm = SecurityManager()

    # Safe inputs
    assert sm.sanitize_input_text("network_partition") == "network_partition"
    assert sm.sanitize_input_text("  cpu_saturation  ") == "cpu_saturation"

    # Script tags rejected
    with pytest.raises(ValueError, match="forbidden HTML/Script markup tags"):
        sm.sanitize_input_text("<script>alert('pwned')</script>")

    with pytest.raises(ValueError, match="forbidden HTML/Script markup tags"):
        sm.sanitize_input_text("<img src=x onerror=alert(1)>")

    # SQL Injection patterns rejected
    with pytest.raises(ValueError, match="forbidden SQL syntax patterns"):
        sm.sanitize_input_text("payment-service'; DROP TABLE services; --")

    with pytest.raises(ValueError, match="forbidden SQL syntax patterns"):
        sm.sanitize_input_text("' UNION SELECT * FROM incidents --")


def test_security_headers_and_rate_limit_headers_in_response():
    client = TestClient(app)
    res = client.get("/api/v1/health")
    assert res.status_code == 200

    # Defensive security headers
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in res.headers

    # Rate limiting headers
    assert "X-RateLimit-Limit" in res.headers
    assert "X-RateLimit-Remaining" in res.headers
    assert "X-RateLimit-Reset" in res.headers
