import pytest
from apps.api.security import SecurityManager, SlidingWindowRateLimiter

def test_sliding_window_rate_limiter_burst_and_sustained():
    limiter = SlidingWindowRateLimiter(requests_per_minute=20, burst_limit=5)
    client_id = "test-client"

    for _ in range(5):
        allowed, retry_after = limiter.is_allowed(client_id)
        assert allowed is True
        assert retry_after == 0

    # Burst rejection
    allowed, retry_after = limiter.is_allowed(client_id)
    assert allowed is False
    assert retry_after >= 1


def test_security_manager_validation():
    sec = SecurityManager()
    sec.auth_enabled = True
    sec.valid_keys = {"test-key-1"}

    assert sec.validate_api_key("test-key-1") is True
    assert sec.validate_api_key("bad-key") is False
    assert sec.validate_api_key(None) is False
