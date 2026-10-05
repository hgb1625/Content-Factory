"""
Common Utilities for Multi-AI Providers (Phase 2)

Provides thread-safe model catalog caching with SHA-256 key fingerprinting,
robust secret sanitization, and standardized HTTP error classification.
"""
import re
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Union
import httpx

from app.services.ai.base import (
    AIProviderError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIBadRequestError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
)

logger = logging.getLogger("app.services.ai.providers")


# ==============================================================================
# SECRET SANITIZATION
# ==============================================================================

_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"gsk_[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"AIzaSy[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"x-api-key:\s*[^\s,;]+", re.IGNORECASE),
    re.compile(r"(?:api_?key|key|token)=([a-zA-Z0-9_\-\.]{8,})", re.IGNORECASE),
    re.compile(r"eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}", re.IGNORECASE),
]


def sanitize_secrets(text: str, *secrets: Optional[str]) -> str:
    """
    Scrub sensitive credentials, tokens, and authorization headers from error messages
    and logging payloads before exposing them to users, logs, or metadata.
    """
    if not text:
        return ""

    sanitized = str(text)

    # 1. Redact specific known secrets
    for s in secrets:
        if s and len(s) >= 4:
            sanitized = sanitized.replace(s, "[REDACTED]")

    # 2. Redact standard credential patterns
    for pat in _SECRET_PATTERNS:
        if "Bearer" in pat.pattern:
            sanitized = pat.sub("Bearer [REDACTED]", sanitized)
        elif "x-api-key" in pat.pattern:
            sanitized = pat.sub("x-api-key: [REDACTED]", sanitized)
        elif "(?:api_?key" in pat.pattern:
            sanitized = pat.sub("key=[REDACTED]", sanitized)
        else:
            sanitized = pat.sub("[REDACTED]", sanitized)

    return sanitized



# ==============================================================================
# MODEL CATALOG CACHE
# ==============================================================================

class ModelCatalogCache:
    """
    In-memory model catalog cache with configurable TTL (default 60 minutes).
    Uses SHA-256 fingerprinting of API keys for cache partitioning, guaranteeing
    that raw credentials are never stored in cache keys or memory dictionaries.
    """

    def __init__(self, ttl_seconds: float = 3600.0):
        self._ttl = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def get_fingerprint(api_key: Optional[str]) -> str:
        """Derive safe, non-reversible SHA-256 hash prefix for cache identity."""
        if not api_key:
            return "no_key"
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    def get(self, provider_id: str, api_key: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached model catalog if present and not expired."""
        fp = self.get_fingerprint(api_key)
        cache_key = f"{provider_id}:{fp}"
        entry = self._cache.get(cache_key)
        if not entry:
            return None

        age = time.time() - entry.get("timestamp", 0.0)
        if age > self._ttl:
            self._cache.pop(cache_key, None)
            return None

        return list(entry.get("models", []))

    def set(self, provider_id: str, api_key: Optional[str], models: List[Dict[str, Any]]) -> None:
        """Store model catalog in cache with current timestamp."""
        fp = self.get_fingerprint(api_key)
        cache_key = f"{provider_id}:{fp}"
        self._cache[cache_key] = {
            "models": list(models),
            "timestamp": time.time()
        }

    def clear(self) -> None:
        """Clear all cached catalogs."""
        self._cache.clear()

    def invalidate_provider(self, provider_id: str) -> None:
        """Invalidate all cached model catalogs for a specific provider."""
        prefix = f"{provider_id.strip().lower()}:"
        to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
        for k in to_delete:
            self._cache.pop(k, None)


# Global cache singleton
global_model_cache = ModelCatalogCache(ttl_seconds=3600.0)


SAFE_DIAGNOSTIC_HEADERS = {
    "retry-after",
    "x-request-id",
    "request-id",
    "cf-ray",
    "traceparent",
    "x-trace-id",
}


def parse_safe_retry_after(raw_val: Optional[Any]) -> Optional[int]:
    """
    Parse Retry-After header safely.
    Returns positive integer seconds (capped at 3600), or None if missing/invalid/negative.
    Never invents or fabricates a wait time.
    """
    if raw_val is None:
        return None
    try:
        val = int(str(raw_val).strip())
        if 0 < val <= 3600:
            return val
    except (ValueError, TypeError):
        pass
    return None


def sanitize_request_id(raw_val: Optional[Any]) -> Optional[str]:
    """
    Normalize and sanitize trace / request ID.
    Allows alphanumeric characters, hyphens, underscores, dots, and colons.
    Caps length to 64 chars. Returns None if invalid or empty.
    """
    if raw_val is None:
        return None
    raw_str = str(raw_val).strip()
    cleaned = "".join(c for c in raw_str if c.isalnum() or c in "-_.:")
    return cleaned[:64] if cleaned else None
 
 
def build_httpx_timeout(
    timeout_val: Any,
    connect_timeout: float = 15.0,
    write_timeout: float = 15.0,
    pool_timeout: float = 10.0
) -> httpx.Timeout:
    """
    Construct explicit httpx.Timeout ensuring bounded connect, write, and pool timeouts
    while honoring requested read timeout.
    """
    if isinstance(timeout_val, httpx.Timeout):
        return timeout_val
    if timeout_val is None:
        return httpx.Timeout(60.0, connect=connect_timeout, write=write_timeout, pool=pool_timeout)
    try:
        t_float = float(timeout_val)
        return httpx.Timeout(
            connect=min(connect_timeout, t_float),
            read=t_float,
            write=min(write_timeout, t_float),
            pool=min(pool_timeout, t_float)
        )
    except (ValueError, TypeError):
        return httpx.Timeout(60.0, connect=connect_timeout, write=write_timeout, pool=pool_timeout)


def classify_http_error(
    status_code: int,
    response_text: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    response_headers: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None
) -> AIProviderError:
    """
    Parse and normalize HTTP error responses into domain AIProviderError exceptions.
    Preserves sanitized upstream messages while ensuring zero secret exposure.
    Captures safe diagnostic metadata (Retry-After, request ID, duration) for 5xx errors.
    """
    upstream_msg = ""
    err_type_str = ""

    if response_text:
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                # Standard OpenAI / Groq / OpenRouter structure
                err_obj = data.get("error", {})
                if isinstance(err_obj, dict):
                    upstream_msg = str(err_obj.get("message", "")).strip()
                    err_type_str = str(err_obj.get("type", "")).strip()
                elif isinstance(err_obj, str):
                    upstream_msg = err_obj.strip()

                # Anthropic structure: {"type": "error", "error": {"type": "...", "message": "..."}}
                if not upstream_msg and "message" in data:
                    upstream_msg = str(data["message"]).strip()
        except Exception:
            upstream_msg = response_text[:300].strip()

    clean_upstream = sanitize_secrets(upstream_msg, api_key)
    error_summary = clean_upstream or f"HTTP {status_code} error from {provider}"

    # 401 Unauthorized / Authentication Error
    if status_code == 401:
        return AIAuthenticationError(
            f"{provider.capitalize()} API 401 Unauthorized: Invalid API key or unauthenticated. {clean_upstream}",
            status_code=401,
            provider=provider,
            model=model,
            upstream_message=clean_upstream
        )

    # 403 Forbidden / Permission Error
    if status_code == 403:
        return AIPermissionError(
            f"{provider.capitalize()} API 403 Forbidden: Permission denied for model '{model}'. {clean_upstream}",
            status_code=403,
            provider=provider,
            model=model,
            upstream_message=clean_upstream
        )

    # 404 Not Found / Model Not Found
    if status_code == 404:
        return AIModelNotFoundError(
            f"{provider.capitalize()} API 404 Not Found: Model '{model}' not found or endpoint invalid. {clean_upstream}",
            status_code=404,
            provider=provider,
            model=model,
            upstream_message=clean_upstream
        )

    # 429 Rate Limit / Quota Exceeded
    if status_code == 429:
        combined_lower = (clean_upstream + " " + err_type_str).lower()
        quota_indicators = [
            "quota",
            "credit",
            "billing",
            "balance",
            "insufficient",
            "exceeded your current quota",
            "daily limit",
            "resource_exhausted"
        ]
        is_quota = any(ind in combined_lower for ind in quota_indicators)

        if is_quota:
            return AIQuotaExceededError(
                f"{provider.capitalize()} API 429 Quota Exceeded: Daily or account credit quota exhausted. {clean_upstream}",
                status_code=429,
                provider=provider,
                model=model,
                upstream_message=clean_upstream
            )
        else:
            return AIRateLimitError(
                f"{provider.capitalize()} API 429 Rate Limit Exceeded: Too many requests. {clean_upstream}",
                status_code=429,
                provider=provider,
                model=model,
                upstream_message=clean_upstream
            )

    # 400 Bad Request
    if status_code == 400:
        return AIBadRequestError(
            f"{provider.capitalize()} API 400 Bad Request: Invalid parameters. {clean_upstream}",
            status_code=400,
            provider=provider,
            model=model,
            upstream_message=clean_upstream
        )

    # 500 / 502 / 503 / 504 Service Unavailable / Server Overload
    if status_code in (500, 502, 503, 504) or status_code >= 500:
        retry_after: Optional[int] = None
        request_id: Optional[str] = None
        if response_headers and isinstance(response_headers, dict):
            lower_headers = {str(k).lower(): v for k, v in response_headers.items() if str(k).lower() in SAFE_DIAGNOSTIC_HEADERS}
            retry_after = parse_safe_retry_after(lower_headers.get("retry-after"))
            raw_req_id = (
                lower_headers.get("x-request-id")
                or lower_headers.get("request-id")
                or lower_headers.get("cf-ray")
                or lower_headers.get("x-trace-id")
                or lower_headers.get("traceparent")
            )
            request_id = sanitize_request_id(raw_req_id)

        return AIServiceUnavailableError(
            f"{provider.capitalize()} API {status_code} Service Unavailable: Upstream server is overloaded. {clean_upstream}",
            status_code=status_code,
            provider=provider,
            model=model,
            upstream_message=clean_upstream,
            retry_after=retry_after,
            request_id=request_id,
            duration_seconds=duration_seconds
        )

    # Fallback generic provider error
    return AIProviderError(
        f"{provider.capitalize()} API Error (HTTP {status_code}): {clean_upstream}",
        status_code=status_code,
        provider=provider,
        model=model,
        upstream_message=clean_upstream,
        error_type=f"http_{status_code}"
    )
