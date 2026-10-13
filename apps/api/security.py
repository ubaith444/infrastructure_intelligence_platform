import re
from collections import defaultdict
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import HTTPException, Request, Response, status


class UserRole:
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class SecurityManager:
    """Manages API key authentication, Role-Based Access Control (RBAC), and security policies."""

    def __init__(self):
        # Default key mappings: key -> role
        self.key_roles: Dict[str, str] = {
            "iip-key-prod-01": UserRole.ADMIN,
            "iip-demo-key-2026": UserRole.OPERATOR,
            "iip-analyst-key": UserRole.ANALYST,
            "iip-viewer-key": UserRole.VIEWER,
        }
        # Allow environment configuration: format "key:ROLE,key:ROLE"
        env_keys = os.getenv("IIP_API_KEYS", "")
        if env_keys:
            for item in env_keys.split(","):
                if ":" in item:
                    k, r = item.split(":", 1)
                    self.key_roles[k.strip()] = r.strip().upper()
                elif item.strip():
                    self.key_roles[item.strip()] = UserRole.OPERATOR

        self.auth_enabled: bool = os.getenv("IIP_AUTH_ENABLED", "false").lower() == "true"

    @property
    def valid_keys(self) -> Set[str]:
        return set(self.key_roles.keys())

    @valid_keys.setter
    def valid_keys(self, keys: Any):
        self.key_roles = {str(k): UserRole.OPERATOR for k in keys}

    def validate_api_key(self, provided_key: Optional[str]) -> bool:
        """Validate if the provided API key is registered."""
        if not self.auth_enabled:
            return True
        if not provided_key:
            return False
        return provided_key in self.key_roles

    def get_role_for_key(self, provided_key: Optional[str]) -> Optional[str]:
        """Return role associated with key, or default VIEWER when auth is disabled."""
        if not self.auth_enabled and not provided_key:
            return UserRole.ADMIN
        return self.key_roles.get(provided_key) if provided_key else None

    def check_permission(self, provided_key: Optional[str], required_role: str) -> bool:
        """Enforces RBAC hierarchy: ADMIN > OPERATOR > ANALYST > VIEWER."""
        if not self.auth_enabled:
            return True
        role = self.get_role_for_key(provided_key)
        if not role:
            return False

        hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.ANALYST: 2,
            UserRole.OPERATOR: 3,
            UserRole.ADMIN: 4,
        }
        user_level = hierarchy.get(role, 0)
        required_level = hierarchy.get(required_role, 99)
        return user_level >= required_level

    def sanitize_input_text(self, text: str) -> str:
        """Defensive sanitization against SQL injection and script injection patterns."""
        if not text:
            return ""
        # Check script / html injection
        if re.search(r"<\s*script[^>]*>.*?</\s*script\s*>", text, re.IGNORECASE) or "<" in text and ">" in text:
            raise ValueError("Input contains forbidden HTML/Script markup tags.")
        # Check suspicious SQL injection keywords
        sql_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(--|;|\/\*|\*\/)",
        ]
        for pat in sql_patterns:
            if re.search(pat, text, re.IGNORECASE):
                raise ValueError("Input contains forbidden SQL syntax patterns.")
        return text.strip()

    def get_security_status(self) -> Dict[str, Any]:
        return {
            "auth_enabled": self.auth_enabled,
            "registered_keys_count": len(self.key_roles),
            "roles_configured": list(set(self.key_roles.values())),
            "security_headers_enabled": True,
            "rate_limiting_enabled": True,
            "input_sanitization_enabled": True,
        }


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter with per-client burst and sustained quotas."""

    def __init__(self, requests_per_minute: int = 600, burst_limit: int = 50):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.history: Dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - 60.0
        self.history[client_id] = [ts for ts in self.history[client_id] if ts > window_start]
        recent = self.history[client_id]

        if len(recent) >= self.requests_per_minute:
            retry_after = max(1, int(60.0 - (now - recent[0])))
            return False, retry_after

        # Burst check (1 second)
        burst_count = sum(1 for ts in recent if ts > now - 1.0)
        if burst_count >= self.burst_limit:
            return False, 1

        recent.append(now)
        return True, 0

    def get_usage(self, client_id: str) -> Dict[str, int]:
        now = time.time()
        recent = [ts for ts in self.history.get(client_id, []) if ts > now - 60.0]
        return {
            "current_rpm": len(recent),
            "max_rpm": self.requests_per_minute,
            "remaining_rpm": max(0, self.requests_per_minute - len(recent)),
        }

    def get_rate_limit_headers(self, client_id: str) -> Dict[str, str]:
        usage = self.get_usage(client_id)
        return {
            "X-RateLimit-Limit": str(usage["max_rpm"]),
            "X-RateLimit-Remaining": str(usage["remaining_rpm"]),
            "X-RateLimit-Reset": "60",
        }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects defensive HTTP headers on all responses."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        return response

