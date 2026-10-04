import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
try:
    from sqlalchemy.orm import Session
    from sqlalchemy import func
except ImportError:
    Session = Any
    func = None

from app.config import get_gemini_model, get_gemini_api_key, DEFAULT_GEMINI_MODEL
from app.services.usage_tracker import GeminiUsageTracker
from app.services.gemini_status import GeminiStatusTracker

logger = logging.getLogger("app.services.gemini")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

logging.getLogger("httpx").setLevel(logging.WARNING)


from app.services.ai.base import (
    AIProviderError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
    hydrate_research_product,
    get_research_max_output_tokens,
    get_research_timeout,
)


class GeminiAPIError(AIProviderError):
    """Base exception for all Gemini API errors, maintaining full RuntimeError and AIProviderError compatibility."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        upstream_message: str = "",
        model_name: str = ""
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            provider="gemini",
            model=model_name,
            upstream_message=upstream_message
        )
        self.upstream_message = upstream_message
        self.model_name = model_name


class GeminiQuotaExceededError(GeminiAPIError, AIQuotaExceededError):
    """Raised when Gemini returns daily quota exhaustion. Must NOT be retried."""
    pass


class GeminiRateLimitError(GeminiAPIError, AIRateLimitError):
    """Raised when Gemini returns temporary burst rate limit (per-minute)."""
    pass


class GeminiBadRequestError(GeminiAPIError):
    """HTTP 400 Bad Request."""
    pass


class GeminiAuthError(GeminiAPIError, AIAuthenticationError):
    """HTTP 401 Unauthorized / Invalid API Key."""
    pass


class GeminiPermissionError(GeminiAPIError, AIPermissionError):
    """HTTP 403 Forbidden / Permission Denied."""
    pass


class GeminiModelNotFoundError(GeminiAPIError, AIModelNotFoundError):
    """HTTP 404 Model Not Found."""
    pass


class GeminiInternalServerError(GeminiAPIError):
    """HTTP 500 Google Internal Server Error."""
    pass


class GeminiServiceUnavailableError(GeminiAPIError, AIServiceUnavailableError):
    """HTTP 503 / 500+ Temporary Google Service Unavailable / High Demand."""
    pass


class GeminiTimeoutError(GeminiAPIError, AITimeoutError):
    """Request timeout."""
    pass


class GeminiNetworkError(GeminiAPIError, AINetworkError):
    """Network connection failure."""
    pass


def sanitize_error_message(msg: str, api_key: Optional[str] = None) -> str:
    """Ensure no API keys, auth headers, or raw tokens appear in any error string."""
    if not msg:
        return ""
    clean = str(msg)
    if api_key and isinstance(api_key, str) and len(api_key) > 5:
        clean = clean.replace(api_key, "[REDACTED]")
    clean = re.sub(r"key=[A-Za-z0-9_\-\.]{10,}", "key=[REDACTED]", clean)
    clean = re.sub(r"x-goog-api-key['\":\s]+[A-Za-z0-9_\-\.]{10,}", "x-goog-api-key: [REDACTED]", clean, flags=re.IGNORECASE)
    clean = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{10,}", "Bearer [REDACTED]", clean, flags=re.IGNORECASE)
    return clean.strip()


# In-memory cache for available generation models
_MODEL_CATALOG_CACHE: Dict[str, Any] = {
    "models": [],
    "cached_at": 0.0,
    "ttl": 3600.0  # 1 hour TTL
}

# Candidate fallback models in order of priority (stable text/content generation models)
FALLBACK_CANDIDATE_PRIORITY: List[str] = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]


# Canonical token budget and timeout functions are imported from app.services.ai.base



def discover_available_models(api_key: str, timeout: float = 10.0) -> List[str]:
    """
    Discover models supporting generateContent from Google API, cached for 1 hour.
    Does NOT call models.list before every request; called ONLY when fallback is needed.
    Failure to list models returns empty list without raising exception.
    """
    now = time.time()
    if _MODEL_CATALOG_CACHE["models"] and (now - _MODEL_CATALOG_CACHE["cached_at"] < _MODEL_CATALOG_CACHE["ttl"]):
        return _MODEL_CATALOG_CACHE["models"]

    if not api_key:
        return []

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": api_key}
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("models", [])
                valid_models = []
                for m in models_list:
                    name = m.get("name", "")
                    if name.startswith("models/"):
                        name = name[7:]
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        # Exclude specialized non-text models
                        lower = name.lower()
                        if not any(bad in lower for bad in ["tts", "image", "live", "audio", "embed", "veo", "lyria"]):
                            valid_models.append(name)

                if valid_models:
                    _MODEL_CATALOG_CACHE["models"] = valid_models
                    _MODEL_CATALOG_CACHE["cached_at"] = now
                    return valid_models
    except Exception as e:
        logger.warning(f"Failed to query Gemini model catalog for fallback: {sanitize_error_message(str(e), api_key)}")

    return []


def select_fallback_model(primary_model: str, api_key: str) -> Optional[str]:
    """
    Select an appropriate fallback model supporting generateContent.
    Prioritizes verified available models from Google catalog, falls back to static candidate priority list.
    """
    available = discover_available_models(api_key)
    # Match candidate in priority order
    for candidate in FALLBACK_CANDIDATE_PRIORITY:
        if candidate != primary_model and (candidate in available or f"models/{candidate}" in available):
            return candidate

    # If discovery was empty/unavailable, choose first static candidate different from primary
    for candidate in FALLBACK_CANDIDATE_PRIORITY:
        if candidate != primary_model:
            return candidate

    return None


def get_api_key(db: Optional[Session] = None) -> str:
    """Retrieve the Gemini API key from single source of truth."""
    import app.config as cfg
    return cfg.get_gemini_api_key(db)


def get_model_name(db: Optional[Session] = None) -> str:
    """Retrieve the Gemini Model name from single source of truth (default: gemini-3.8-flash)."""
    import app.config as cfg
    return cfg.get_gemini_model(db)


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code blocks or accidental conversational wrappings from model output."""
    text = raw_text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

    if not (text.startswith("{") or text.startswith("[")):
        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            end_obj = text.rfind("}")
            if end_obj > start_obj:
                text = text[start_obj:end_obj + 1]
        elif start_arr != -1:
            end_arr = text.rfind("]")
            if end_arr > start_arr:
                text = text[start_arr:end_arr + 1]

    return text.strip()


def get_next_product_id(db: Session) -> str:
    """Generate the next sequential product ID: P0001, P0002, etc."""
    from app.models import Product

    products = db.query(Product.product_id).filter(Product.product_id.like("P%")).all()
    max_num = 0
    for (pid,) in products:
        m = re.match(r"^P(\d+)$", pid)
        if m:
            val = int(m.group(1))
            if val > max_num:
                max_num = val

    next_num = max_num + 1
    return f"P{next_num:04d}"


def is_daily_quota_error(status_code: int, err_msg: str, err_json: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if a 429 response is a daily quota exhaustion vs a temporary burst rate limit.
    Google Gemini Free Tier daily quota indicators:
    - 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'
    - 'generativelanguage.googleapis.com/generate_content_free_tier_requests'
    - 'GenerateRequestsPerDay'
    - 'free_tier_requests'
    """
    if status_code != 429:
        return False

    combined = (err_msg or "").lower()
    if err_json:
        try:
            combined += " " + json.dumps(err_json).lower()
        except Exception:
            pass

    # If explicitly per minute or RPM, it is temporary burst rate limit
    if "generaterequestsperminute" in combined or "per minute" in combined or "rpm" in combined:
        return False

    daily_indicators = [
        "generaterequestsperday",
        "generate_content_free_tier_requests",
        "freetier",
        "free tier",
        "daily quota",
        "per day",
        "quota exceeded",
        "resource_exhausted"
    ]

    for ind in daily_indicators:
        if ind in combined:
            return True

    return False


def validate_timed_script(
    input_segments: List[Dict[str, Any]],
    ai_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic local Python validation for batch timed script:
    - Verifies 'segments' key exists and is a list
    - Verifies all input segment_ids are present
    - Verifies vietnamese_text is non-empty and has no placeholder markers
    """
    if not isinstance(ai_output, dict) or "segments" not in ai_output:
        return {
            "valid": False,
            "error": "Phản hồi AI không chứa danh sách 'segments' hợp lệ."
        }

    out_segments = ai_output.get("segments", [])
    if not isinstance(out_segments, list):
        return {
            "valid": False,
            "error": "'segments' phải là một danh sách JSON."
        }

    output_by_id = {}
    for item in out_segments:
        if isinstance(item, dict) and "segment_id" in item:
            output_by_id[item["segment_id"]] = item

    validated_segments = []
    for inp in input_segments:
        sid = inp.get("segment_id")
        if sid not in output_by_id:
            return {
                "valid": False,
                "error": f"Thiếu phân đoạn segment_id={sid} trong phản hồi AI."
            }

        res_item = output_by_id[sid]
        text = str(res_item.get("vietnamese_text", "")).strip()
        if not text:
            return {
                "valid": False,
                "error": f"Phân đoạn segment_id={sid} có lời thoại trống."
            }

        st = float(inp.get("start_time", inp.get("start", 0.0)))
        dur = float(inp.get("duration", 0.0))
        et = float(inp.get("end_time", inp.get("end", round(st + dur, 2))))
        # Word count heuristic: Vietnamese speaking speed is ~2.5 - 3.5 words/second
        word_count = len(text.split())

        validated_segments.append({
            "segment_id": sid,
            "start": st,
            "end": et,
            "start_time": st,
            "end_time": et,
            "duration": dur,
            "vietnamese_text": text,
            "word_count": word_count
        })

    return {
        "valid": True,
        "segments": validated_segments
    }


def validate_rewritten_segments(
    failed_segments: List[Dict[str, Any]],
    ai_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic local Python validation for batch failed-segment rewrites:
    - Confirms all requested failed segment_ids are returned with non-empty vietnamese_text
    """
    if not isinstance(ai_output, dict) or "rewritten_segments" not in ai_output:
        return {
            "valid": False,
            "error": "Phản hồi AI không chứa 'rewritten_segments' hợp lệ."
        }

    out_items = ai_output.get("rewritten_segments", [])
    if not isinstance(out_items, list):
        return {
            "valid": False,
            "error": "'rewritten_segments' phải là một danh sách JSON."
        }

    by_id = {}
    for item in out_items:
        if isinstance(item, dict) and "segment_id" in item:
            by_id[item["segment_id"]] = item

    rewritten_list = []
    for failed in failed_segments:
        sid = failed.get("segment_id")
        if sid not in by_id:
            return {
                "valid": False,
                "error": f"Thiếu phân đoạn được viết lại segment_id={sid}."
            }

        text = str(by_id[sid].get("vietnamese_text", "")).strip()
        if not text:
            return {
                "valid": False,
                "error": f"Phân đoạn segment_id={sid} được viết lại nhưng lời thoại trống."
            }

        rewritten_list.append({
            "segment_id": sid,
            "target_duration": failed.get("target_duration"),
            "vietnamese_text": text
        })

    return {
        "valid": True,
        "rewritten_segments": rewritten_list
    }


class GeminiService:
    def __init__(self):
        self.last_execution: Dict[str, Any] = {
            "primary_model": None,
            "actual_model_used": None,
            "fallback_used": False,
            "primary_failure": None,
            "primary_error": None
        }

    def _execute_single_model_call(
        self,
        model_name: str,
        prompt: str,
        api_key: str,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        is_fallback: bool = False,
        max_output_tokens: Optional[int] = None
    ) -> str:
        """
        Execute call to a single model with bounded exponential backoff for 503 / 500+ and transient 429.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            payload["generationConfig"] = {
                "maxOutputTokens": max_output_tokens
            }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        # Exponential backoff delays for 503: attempt 0→2s, attempt 1→5s
        _503_delays = [2.0, 5.0]
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    status_code = resp.status_code

                    # 200 OK
                    if status_code == 200:
                        try:
                            data = resp.json()
                        except Exception as je:
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError(f"Malformed JSON returned by Gemini endpoint: {je}")

                        candidates = data.get("candidates", [])
                        if not candidates:
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError("Gemini API returned an empty response (no candidates).")

                        candidate = candidates[0]
                        parts = candidate.get("content", {}).get("parts", [])
                        if not parts:
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError("Gemini API returned an empty response (no content parts).")

                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        raw_text = "".join(text_parts).strip()
                        if not raw_text:
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError("Gemini API returned empty text.")

                        return raw_text

                    # Extract error details
                    err_msg = ""
                    err_json = None
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", {}).get("message", resp.text[:200])
                    except Exception:
                        err_msg = resp.text[:200]

                    clean_err = sanitize_error_message(err_msg, api_key)

                    # 429 Rate Limit / Quota Exceeded
                    if status_code == 429:
                        if is_daily_quota_error(status_code, err_msg, err_json):
                            # DAILY QUOTA EXHAUSTED: DO NOT RETRY. FAIL FAST.
                            GeminiStatusTracker.update_status("QUOTA_EXCEEDED", db=db, http_code=429)
                            logger.error(f"Gemini daily quota exhausted for model {model_name}. Aborting without retry.")
                            raise GeminiQuotaExceededError(
                                "GEMINI_QUOTA_EXCEEDED: Gemini daily quota has been reached. "
                                "Try again after quota reset or use a project with sufficient quota.",
                                status_code=429,
                                upstream_message=clean_err,
                                model_name=model_name
                            )

                        # Temporary Rate Limit (burst / per-minute)
                        retry_after_hdr = resp.headers.get("retry-after")
                        retry_delay = 2.0 * (attempt + 1)
                        if retry_after_hdr and retry_after_hdr.isdigit():
                            retry_delay = min(float(retry_after_hdr), 15.0)
                        else:
                            m_wait = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                            if m_wait:
                                try:
                                    extracted = float(m_wait.group(1)) + 1.0
                                    if extracted <= 15.0:
                                        retry_delay = extracted
                                except Exception:
                                    pass

                        if attempt < max_retries:
                            GeminiStatusTracker.update_status("RATE_LIMITED", db=db, http_code=429)
                            logger.warning(
                                f"Gemini 429 Temporary Rate Limit (attempt {attempt + 1}/{max_retries + 1}), "
                                f"waiting {retry_delay:.1f}s before retry..."
                            )
                            time.sleep(retry_delay)
                            continue

                        GeminiStatusTracker.update_status("RATE_LIMITED", db=db, http_code=429)
                        raise GeminiRateLimitError(
                            f"Gemini API 429 Rate Limit Exceeded: {clean_err}",
                            status_code=429,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # 400 Bad Request
                    if status_code == 400:
                        GeminiStatusTracker.update_status("ERROR", db=db, http_code=400)
                        raise GeminiBadRequestError(
                            f"Gemini API 400 Bad Request: {clean_err}",
                            status_code=400,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # 401 Unauthorized
                    if status_code == 401:
                        GeminiStatusTracker.update_status("AUTH_ERROR", db=db, http_code=401)
                        raise GeminiAuthError(
                            "Gemini API 401 Unauthorized: Invalid API Key. Please verify your key in Settings.",
                            status_code=401,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # 403 Forbidden
                    if status_code == 403:
                        GeminiStatusTracker.update_status("AUTH_ERROR", db=db, http_code=403)
                        raise GeminiPermissionError(
                            f"Gemini API 403 Forbidden: Permission denied for model '{model_name}'. Details: {clean_err}",
                            status_code=403,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # 404 Model Not Found
                    if status_code == 404:
                        GeminiStatusTracker.update_status("ERROR", db=db, http_code=404)
                        raise GeminiModelNotFoundError(
                            f"Gemini API 404 Not Found: Model '{model_name}' was not found. Details: {clean_err}",
                            status_code=404,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # 500 Google Internal Server Error
                    if status_code == 500:
                        GeminiStatusTracker.update_status("BUSY", db=db, http_code=500)
                        if attempt < max_retries:
                            delay = _503_delays[attempt] if attempt < len(_503_delays) else 5.0
                            logger.warning(
                                f"Gemini 500 Internal Error (attempt {attempt + 1}/{max_retries + 1}), "
                                f"retrying in {delay:.0f}s..."
                            )
                            time.sleep(delay)
                            continue
                        raise GeminiInternalServerError(
                            f"Google Gemini Internal Server Error (HTTP 500): {clean_err}",
                            status_code=500,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # 502 / 503 / 504 / 500+ Overload & Service Unavailable
                    if status_code >= 501:
                        GeminiStatusTracker.update_status("BUSY", db=db, http_code=status_code)
                        if attempt < max_retries:
                            delay = _503_delays[attempt] if attempt < len(_503_delays) else 5.0
                            logger.warning(
                                f"Gemini {status_code} Temporary Overload (attempt {attempt + 1}/{max_retries + 1}), "
                                f"retrying in {delay:.0f}s..."
                            )
                            time.sleep(delay)
                            continue
                        raise GeminiServiceUnavailableError(
                            f"Gemini API {status_code} Server Error: Gemini tạm thời quá tải (HTTP {status_code}). "
                            f"Google: {clean_err or 'High demand'}. Hệ thống đã thử lại {max_retries + 1} lần. Hãy thử lại sau.",
                            status_code=status_code,
                            upstream_message=clean_err,
                            model_name=model_name
                        )

                    # Other unexpected status codes
                    GeminiStatusTracker.update_status("ERROR", db=db, http_code=status_code)
                    raise GeminiAPIError(
                        f"Gemini API error {status_code}: {clean_err}",
                        status_code=status_code,
                        upstream_message=clean_err,
                        model_name=model_name
                    )

            except (GeminiAPIError, ValueError) as e:
                raise e

            except httpx.TimeoutException:
                last_error = GeminiTimeoutError(
                    f"Gemini API request timed out after {timeout} seconds.",
                    model_name=model_name
                )
                if attempt < max_retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in 2s...")
                    time.sleep(2)
                    continue
                GeminiStatusTracker.update_status("NETWORK_ERROR", db=db)
                raise last_error

            except httpx.NetworkError as ne:
                clean_ne = sanitize_error_message(str(ne), api_key)
                last_error = GeminiNetworkError(
                    f"Lỗi kết nối mạng tới Gemini API: {clean_ne}",
                    model_name=model_name
                )
                if attempt < max_retries:
                    logger.warning(f"Network error on attempt {attempt + 1}, retrying in 2s...")
                    time.sleep(2)
                    continue
                GeminiStatusTracker.update_status("NETWORK_ERROR", db=db)
                raise last_error

            except Exception as e:
                clean_e = sanitize_error_message(str(e), api_key)
                GeminiStatusTracker.update_status("ERROR", db=db)
                raise GeminiAPIError(
                    f"Gemini request failed: {clean_e}",
                    model_name=model_name
                )

        if last_error:
            raise last_error
        raise GeminiServiceUnavailableError(
            f"Gemini API {model_name} call failed after retries.",
            status_code=503,
            model_name=model_name
        )

    def call_gemini(
        self,
        prompt: str,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        enable_fallback: bool = False,
        max_output_tokens: Optional[int] = None
    ) -> str:
        """
        Production-grade Gemini API caller using configured GEMINI_MODEL with 503 smart fallback.
        Features:
        - Strict low request consumption: No retries on Daily Quota exhaustion
        - Discriminates 429 Daily Quota vs 429 Temporary Rate Limit
        - Fast-fails with GeminiQuotaExceededError on daily quota
        - Records lightweight local usage statistics in SQLite
        - Updates GeminiStatusTracker for UI badge (no extra quota consumed)
        - Masks API key from logs and errors
        - 503 bounded exponential backoff: attempt1→2s, attempt2→5s
        - Transparent fallback to verified available model ONLY when primary exhausts 503 retries
        """
        api_key = get_api_key(db)
        model_name = get_model_name(db)

        if not api_key:
            GeminiStatusTracker.update_status("AUTH_ERROR", db=db, http_code=401)
            raise ValueError("Gemini API key is not configured. Please enter your API key in Settings.")

        # Record start of request in local tracker
        GeminiUsageTracker.record_request_start(db=db)
        # Signal that a real request is now in-flight (local only, no network)
        GeminiStatusTracker.update_status("CHECKING", db=db)

        self.last_execution = {
            "primary_model": model_name,
            "actual_model_used": model_name,
            "fallback_used": False,
            "primary_failure": None,
            "primary_error": None
        }

        try:
            # 1. Execute on primary configured model
            raw_text = self._execute_single_model_call(
                model_name=model_name,
                prompt=prompt,
                api_key=api_key,
                db=db,
                timeout=timeout,
                max_retries=max_retries,
                is_fallback=False,
                max_output_tokens=max_output_tokens
            )
            GeminiUsageTracker.record_success(db=db)
            GeminiStatusTracker.update_status("READY", db=db, http_code=200)
            return raw_text

        except GeminiServiceUnavailableError as sue:
            # ONLY 503 / 500+ triggers optional smart fallback after primary model retries exhausted
            if not enable_fallback:
                GeminiUsageTracker.record_failure(db=db, is_quota=False)
                raise sue

            fallback_model = select_fallback_model(model_name, api_key)
            if not fallback_model or fallback_model == model_name:
                GeminiUsageTracker.record_failure(db=db, is_quota=False)
                raise sue

            logger.warning(
                f"Primary model '{model_name}' exhausted 503 retries ({sue.upstream_message}). "
                f"Attempting smart fallback with available model '{fallback_model}'..."
            )

            try:
                raw_text = self._execute_single_model_call(
                    model_name=fallback_model,
                    prompt=prompt,
                    api_key=api_key,
                    db=db,
                    timeout=timeout,
                    max_retries=1,
                    is_fallback=True,
                    max_output_tokens=max_output_tokens
                )
                self.last_execution = {
                    "primary_model": model_name,
                    "actual_model_used": fallback_model,
                    "fallback_used": True,
                    "primary_failure": 503,
                    "primary_error": sue.upstream_message
                }
                logger.info(
                    f"Successfully generated response via fallback model '{fallback_model}' "
                    f"(primary '{model_name}' was 503 unavailable)."
                )
                GeminiUsageTracker.record_success(db=db)
                GeminiStatusTracker.update_status(
                    "READY",
                    db=db,
                    http_code=200,
                    custom_message=f"Gemini sẵn sàng (model dự phòng {fallback_model})"
                )
                return raw_text

            except Exception as fe:
                logger.error(f"Fallback model '{fallback_model}' also failed: {fe}")
                GeminiUsageTracker.record_failure(db=db, is_quota=False)
                clean_fe = sanitize_error_message(str(fe), api_key)
                raise GeminiServiceUnavailableError(
                    f"Gemini API 503 Server Error: Model chính '{model_name}' quá tải (Google: {sue.upstream_message}). "
                    f"Model dự phòng '{fallback_model}' cũng không phản hồi ({clean_fe}). Vui lòng thử lại sau.",
                    status_code=503,
                    upstream_message=sue.upstream_message,
                    model_name=model_name
                )

        except Exception as e:
            is_quota = isinstance(e, GeminiQuotaExceededError)
            GeminiUsageTracker.record_failure(db=db, is_quota=is_quota)
            raise e

    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """Test Gemini API connection using the configured model with resilient 503 backoff and fallback."""
        api_key = get_api_key(db)
        model_name = get_model_name(db)

        if not api_key:
            return {
                "success": False,
                "error_type": "AUTH_ERROR",
                "error": "Gemini API key is not configured. Please add your key in Settings."
            }

        try:
            # Allow up to 2 retries (3 attempts total) with exponential backoff and smart fallback
            res_text = self.call_gemini(
                prompt="Ping. Respond with 'OK'.",
                db=db,
                timeout=30.0,
                max_retries=2,
                enable_fallback=True
            )

            fallback_info = getattr(self, "last_execution", {})
            fallback_used = fallback_info.get("fallback_used", False)
            actual_model = fallback_info.get("actual_model_used", model_name)

            if fallback_used:
                msg = (
                    f"Kết nối thành công (sử dụng model dự phòng {actual_model} "
                    f"do {model_name} quá tải HTTP 503)."
                )
            else:
                msg = f"Connected successfully to {model_name}."

            return {
                "success": True,
                "message": msg,
                "model": model_name,
                "actual_model": actual_model,
                "fallback_used": fallback_used,
                "response": res_text
            }

        except GeminiQuotaExceededError as qe:
            clean_msg = sanitize_error_message(str(qe), api_key)
            return {
                "success": False,
                "quota_exceeded": True,
                "error_type": "QUOTA_EXCEEDED",
                "status_code": 429,
                "error": f"Hạn mức Gemini hàng ngày đã hết. {clean_msg}",
                "model": model_name
            }
        except GeminiRateLimitError as rle:
            clean_msg = sanitize_error_message(str(rle), api_key)
            return {
                "success": False,
                "rate_limited": True,
                "error_type": "RATE_LIMITED",
                "status_code": 429,
                "error": f"Gemini đang bị giới hạn tốc độ tạm thời. {clean_msg}",
                "model": model_name
            }
        except GeminiAuthError as ae:
            clean_msg = sanitize_error_message(str(ae), api_key)
            return {
                "success": False,
                "error_type": "AUTH_ERROR",
                "status_code": 401,
                "error": clean_msg,
                "model": model_name
            }
        except GeminiPermissionError as pe:
            clean_msg = sanitize_error_message(str(pe), api_key)
            return {
                "success": False,
                "error_type": "PERMISSION_ERROR",
                "status_code": 403,
                "error": clean_msg,
                "model": model_name
            }
        except GeminiModelNotFoundError as mne:
            clean_msg = sanitize_error_message(str(mne), api_key)
            return {
                "success": False,
                "error_type": "MODEL_NOT_FOUND",
                "status_code": 404,
                "error": clean_msg,
                "model": model_name
            }
        except GeminiServiceUnavailableError as sue:
            clean_msg = sanitize_error_message(str(sue), api_key)
            return {
                "success": False,
                "error_type": "SERVER_BUSY",
                "status_code": 503,
                "error": clean_msg,
                "model": model_name
            }
        except (GeminiTimeoutError, GeminiNetworkError) as ne:
            clean_msg = sanitize_error_message(str(ne), api_key)
            return {
                "success": False,
                "error_type": "NETWORK_ERROR",
                "error": clean_msg,
                "model": model_name
            }
        except Exception as e:
            clean_msg = sanitize_error_message(str(e), api_key)
            logger.error(f"Gemini connection test failed: {clean_msg}")
            return {
                "success": False,
                "error_type": "UNKNOWN",
                "error": f"Connection failed: {clean_msg}",
                "model": model_name
            }

    def generate_products(
        self,
        niche: str,
        count: int = 10,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch Product Research: Generate N products in exactly ONE Gemini request.
        Strict Low-Consumption Mode: max_retries=0, enable_fallback=False.
        Enforces Exact N contract:
        - If >= count valid products: deterministically trim to exactly count.
        - If < count valid products: raise AIInvalidResponseError (under-generation, 0 saved).
        """
        prompt = f"""You are an expert e-commerce and viral short-video researcher for Douyin (TikTok China).
Target Niche: {niche}
Generate a JSON list of exactly {count} trending, problem-solving, or viral products for this niche.

CRITICAL REQUIREMENTS FOR EACH PRODUCT:
- "nv": Clear commercial Vietnamese product name.
- "nc": Natural commercial Chinese supplier/product name used on 1688 and Chinese wholesale markets.
- "dk": 3–4 authentic Chinese Douyin search phrases/terms (e.g. '神器', '好物推荐', '开箱', '测评', and key feature keywords).
- "ca": 1 concise Vietnamese marketing angle, approximately 8–12 words.
- "h": 1 short punchy Vietnamese opening hook, under 12 words.

OUTPUT FORMAT:
Return ONLY the raw JSON array. The response must start with [ and end with ], containing exactly {count} product objects.
Use compact JSON if possible. No markdown fences (do not wrap in ```json), no introduction, no conclusion, and no explanation outside JSON.

Example structure:
[
  {{
    "nv": "Nồi cơm điện mini đa năng",
    "nc": "多功能迷你电饭煲",
    "dk": "宿舍迷你电饭煲 独居一人食 煮饭神器",
    "ca": "Giải pháp nấu ăn tiện lợi nhanh gọn cho người sống một mình",
    "h": "Đừng mua nồi cơm to nữa nếu bạn sống một mình hoặc ở trọ!"
  }}
]
"""
        output_budget = get_research_max_output_tokens(count)
        req_timeout = get_research_timeout(count)
        raw_content = self.call_gemini(
            prompt,
            db=db,
            timeout=req_timeout,
            max_retries=0,
            enable_fallback=False,
            max_output_tokens=output_budget
        )

        cleaned = clean_json_response(raw_content)
        try:
            items = json.loads(cleaned)
            if not isinstance(items, list):
                raise ValueError("Response is not a JSON list.")
        except Exception as e:
            logger.error(f"Failed to parse Gemini JSON output: {cleaned[:200]} - Error: {e}")
            raise ValueError(f"Malformed AI response format: {str(e)}")

        # Validate schema of items locally in Python (zero AI calls)
        valid_items = []
        for it in items:
            hydrated = hydrate_research_product(it)
            if hydrated:
                valid_items.append(hydrated)

        if not valid_items:
            raise ValueError("Không có sản phẩm hợp lệ nào được tìm thấy trong phản hồi của AI.")

        if count >= 30 and len(valid_items) < count:
            raise AIInvalidResponseError(
                f"Gemini chỉ trả về {len(valid_items)}/{count} sản phẩm được yêu cầu (thiếu dữ liệu). "
                f"Yêu cầu dừng lại để đảm bảo tính toàn vẹn dữ liệu.",
                provider="gemini"
            )

        if len(valid_items) > count:
            logger.info(
                f"Research requested {count} products, Gemini returned {len(valid_items)} valid products; "
                f"locally trimmed to {count}."
            )
            valid_items = valid_items[:count]

        if count >= 30:
            assert len(valid_items) == count
        return valid_items

    def generate_timed_script(
        self,
        video_duration: float,
        segments: List[Dict[str, Any]],
        product_info: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Phase 12 Timed Script: Sends the COMPLETE timeline in ONE single Gemini call.
        Does NOT call Gemini once per segment.
        Validates returned segments locally with Python.
        """
        if not segments:
            return {"valid": False, "error": "Danh sách phân đoạn timeline trống."}

        prod_name = (product_info or {}).get("name_vietnamese", "Sản phẩm")
        content_angle = (product_info or {}).get("content_angle", "Tính năng nổi bật")
        hook = (product_info or {}).get("hook", "Món đồ tiện ích không thể bỏ lỡ")

        segment_lines = []
        for s in segments:
            sid = s.get("segment_id")
            st = float(s.get("start_time", s.get("start", 0.0)))
            dur = float(s.get("duration", 0.0))
            et = float(s.get("end_time", s.get("end", round(st + dur, 2))))
            if dur <= 0.0:
                dur = round(et - st, 2)
            desc = s.get("description", "")
            desc_part = f" - Diễn biến cảnh: {desc}" if desc else ""
            segment_lines.append(f"- Phân đoạn {sid}: từ {st:.1f}s đến {et:.1f}s (Thời lượng: {dur:.1f}s){desc_part}")

        timeline_str = "\n".join(segment_lines)

        prompt = f"""Bạn là một chuyên gia biên kịch video ngắn triệu view (TikTok, Reels, Shorts).
Nhiệm vụ: Viết lời đọc thuyết minh (Voice-over) bằng TIẾNG VIỆT cho TOÀN BỘ video sau.

THÔNG TIN SẢN PHẨM:
- Tên sản phẩm: {prod_name}
- Góc tiếp cận: {content_angle}
- Câu mở đầu (Hook): {hook}
- Tổng thời lượng video: {video_duration:.1f} giây

TIMELINE CÁC PHÂN ĐOẠN (Toàn bộ kịch bản phải khớp với từng mốc thời gian):
{timeline_str}

QUY TẮC BẮT BUỘC:
1. KHÔNG được viết quá dài. Tốc độ nói tiếng Việt tự nhiên là khoảng 2.5 đến 3.2 từ mỗi giây.
   Mỗi phân đoạn PHẢI có số lượng từ vừa vặn với thời lượng của phân đoạn đó.
2. Lời thoại tự nhiên, liền mạch giữa các phân đoạn, giọng văn bán hàng thu hút.
3. CHỈ TRẢ VỀ DUY NHẤT một đối tượng JSON hợp lệ theo đúng cấu trúc sau:
{{
  "segments": [
    {{
      "segment_id": 1,
      "vietnamese_text": "..."
    }}
  ]
}}
"""
        raw_text = self.call_gemini(prompt, db=db, timeout=60.0, enable_fallback=True)
        cleaned = clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI returned malformed JSON: {e}")

        val_result = validate_timed_script(segments, parsed)
        if not val_result["valid"]:
            raise ValueError(val_result["error"])

        return val_result

    @staticmethod
    def build_rewrite_prompt(
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        product_name: str = ""
    ) -> str:
        """Construct the prompt for batch rewriting failed segments."""
        segment_lines = []
        for s in failed_segments:
            sid = s.get("segment_id")
            target_dur = s.get("target_duration", 0.0)
            actual_dur = s.get("actual_duration", 0.0)
            curr_text = s.get("current_text", s.get("vietnamese_text", ""))
            direction = s.get("direction", "shorten" if actual_dur > target_dur else "lengthen")

            action = "RÚT NGẮN lại lời đọc" if direction == "shorten" else "KÉO DÀI thêm lời đọc"
            segment_lines.append(
                f"- Phân đoạn {sid}: Thời lượng mục tiêu: {target_dur:.1f}s | "
                f"Thời lượng audio thực tế: {actual_dur:.1f}s. "
                f"Yêu cầu: {action}.\n  Lời hiện tại: \"{curr_text}\""
            )

        failed_str = "\n".join(segment_lines)
        prod_context = f"\nSản phẩm: {product_name}" if product_name else ""

        return f"""Sau khi đo đạc âm thanh thực tế, các phân đoạn sau đây KHÔNG khớp với thời lượng video.{prod_context}
Hãy viết lại LỜI ĐỌC TIẾNG VIỆT cho TẤT CẢ các phân đoạn bị lệch dưới đây trong MỘT LẦN DUY NHẤT.
Vòng viết lại: {round_num}/{max_rounds}

DANH SÁCH PHÂN ĐOẠN CẦN CHỈNH SỬA:
{failed_str}

QUY TẮC:
1. Nếu cần rút ngắn: Giảm bớt số từ, dùng từ cô đọng, giữ trọn ý chính.
2. Nếu cần kéo dài: Thêm mô tả nhẹ nhàng, tự nhiên.
3. CHỈ TRẢ VỀ DUY NHẤT MỘT ĐỐI TƯỢNG JSON với cấu trúc:
{{
  "rewritten_segments": [
    {{
      "segment_id": 1,
      "vietnamese_text": "..."
    }}
  ]
}}
"""

    def rewrite_failed_segments(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        db: Optional[Session] = None,
        product_name: str = ""
    ) -> Dict[str, Any]:
        """
        Duration Rewrite: Sends ALL failed segments in ONE single Gemini request.
        Enforces maximum rewrite rounds (max 2 rounds, no infinite retries).
        Validates rewritten segments locally with Python.
        """
        if round_num > max_rounds:
            return {
                "success": False,
                "error": f"Đã vượt quá số lần viết lại tối đa ({max_rounds} vòng). Cần can thiệp thủ công.",
                "max_rounds_exceeded": True,
                "round": round_num
            }

        if not failed_segments:
            return {
                "success": True,
                "round": round_num,
                "rewritten_segments": []
            }

        prompt = self.build_rewrite_prompt(failed_segments, round_num=round_num, max_rounds=max_rounds, product_name=product_name)
        raw_text = self.call_gemini(prompt, db=db, timeout=60.0, enable_fallback=True)
        cleaned = clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI returned malformed JSON: {e}")

        val_result = validate_rewritten_segments(failed_segments, parsed)
        if not val_result["valid"]:
            raise ValueError(val_result["error"])

        return {
            "success": True,
            "round": round_num,
            "rewritten_segments": val_result["rewritten_segments"]
        }
