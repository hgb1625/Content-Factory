"""
MWAPI Gateway Provider Adapter for Multi-AI Architecture (Phase 6)

Implements Cloud AI Gateway support via MWAPI (Sub2API Subscription Conversion Platform):
- Base: https://api.mwapi.dev
- Generation: POST /v1/chat/completions (OpenAI-compatible)
- Model Discovery: GET /v1/models (0 generation tokens)
- Test Connection: GET /v1/models (0 generation tokens)
- Gateway Routing Safety:
  * Strict single-model submission.
  * Preserves customer-selected model without implicit discovery during generation.
  * Exposes provider_type = 'gateway'.
"""
import time
import json
import logging
from typing import Optional, Dict, Any, List
import httpx
from sqlalchemy.orm import Session

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIAuthenticationError,
    AIPermissionError,
    AIBadRequestError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
)
from app.services.ai.providers.common import (
    sanitize_secrets,
    global_model_cache,
    classify_http_error,
    build_httpx_timeout,
)
from app.config import get_mwapi_api_key, get_mwapi_model, normalize_model_name

logger = logging.getLogger("app.services.ai.providers.mwapi")


class MWAPIProvider(AIProvider):
    """Cloud AI Gateway Provider for MWAPI (Sub2API Relay)."""

    BASE_URL = "https://api.mwapi.dev"

    def __init__(self):
        self._last_execution: Optional[ExecutionMetadata] = None

    @property
    def provider_id(self) -> str:
        return "mwapi"

    @property
    def display_name(self) -> str:
        return "MWAPI Gateway"

    @property
    def provider_type(self) -> str:
        return "gateway"

    def generate(
        self,
        prompt: str,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        enable_fallback: bool = True,
        options: Optional[AIGenerationOptions] = None,
        **kwargs
    ) -> str:
        """
        Generate content using MWAPI Chat Completions endpoint.
        Strictly obeys max_retries (max_retries=0 for Research => exactly 1 HTTP call).
        Strictly submits EXACTLY ONE model string.
        """
        local_session = None
        target_db = db
        if target_db is None:
            try:
                from app.database import SessionLocal
                if SessionLocal:
                    local_session = SessionLocal()
                    target_db = local_session
            except Exception as e:
                logger.debug(f"Could not open local DB session in MWAPIProvider: {e}")

        try:
            api_key = get_mwapi_api_key(target_db)
            raw_model = kwargs.get("model") or get_mwapi_model(target_db)
            model_name = normalize_model_name(self.provider_id, raw_model)
        finally:
            if local_session is not None:
                try:
                    local_session.close()
                except Exception:
                    pass

        max_output_tokens = kwargs.get("max_output_tokens")
        temperature = kwargs.get("temperature")
        system_instruction = kwargs.get("system_instruction")

        if options is not None:
            timeout = options.timeout
            max_retries = options.max_retries
            enable_fallback = options.enable_fallback
            if options.max_output_tokens is not None:
                max_output_tokens = options.max_output_tokens
            if options.temperature is not None:
                temperature = options.temperature
            if options.system_instruction is not None:
                system_instruction = options.system_instruction
            if options.extra_params and "httpx_timeout" in options.extra_params:
                timeout = options.extra_params["httpx_timeout"]

        if not api_key:
            err = AIAuthenticationError(
                "MWAPI API key is not configured. Please add your key in Settings or .env.",
                provider=self.provider_id,
                model=model_name
            )
            self._last_execution = ExecutionMetadata(
                provider=self.provider_id,
                provider_type=self.provider_type,
                configured_model=model_name,
                actual_model_used="",
                status="FAILED",
                error_type="not_configured",
                attempts=0
            )
            raise err

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            payload["max_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ContentFactory/1.0"
        }

        url = f"{self.BASE_URL}/v1/chat/completions"
        total_attempts = max_retries + 1
        start_time = time.time()
        client_timeout = build_httpx_timeout(timeout)

        for attempt in range(total_attempts):
            try:
                with httpx.Client(timeout=client_timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception as je:
                        raise AIInvalidResponseError(
                            f"MWAPI returned unparseable JSON: {je}",
                            provider=self.provider_id,
                            model=model_name
                        )

                    choices = data.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        raise AIInvalidResponseError(
                            "MWAPI response contains no choices.",
                            provider=self.provider_id,
                            model=model_name
                        )

                    choice_0 = choices[0]
                    finish_reason = choice_0.get("finish_reason")
                    msg_obj = choice_0.get("message", {})
                    content = msg_obj.get("content", "")
                    if content is None:
                        content = ""

                    usage = data.get("usage", {})
                    in_tok = usage.get("prompt_tokens")
                    out_tok = usage.get("completion_tokens")
                    tot_tok = usage.get("total_tokens")

                    actual_model = data.get("model", model_name)
                    dur = round(time.time() - start_time, 3)

                    self._last_execution = ExecutionMetadata(
                        provider=self.provider_id,
                        provider_type=self.provider_type,
                        configured_model=model_name,
                        actual_model_used=actual_model,
                        fallback_used=False,
                        status="SUCCESS",
                        attempts=attempt + 1,
                        duration_seconds=dur,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        total_tokens=tot_tok,
                        finish_reason=finish_reason
                    )

                    if finish_reason == "length":
                        logger.warning(
                            f"[RESEARCH TOKEN LIMIT REACHED] Provider={self.provider_id}, Model={model_name}, "
                            f"OutputTokens={out_tok}, MaxTokens={max_output_tokens}"
                        )

                    return content

                dur = round(time.time() - start_time, 3)
                err = classify_http_error(
                    status_code=resp.status_code,
                    response_text=resp.text,
                    provider=self.provider_id,
                    model=model_name,
                    api_key=api_key,
                    response_headers=dict(resp.headers),
                    duration_seconds=dur
                )

                # Retry on 500/502/503/504 if attempts remain
                if resp.status_code in (500, 502, 503, 504) and attempt < total_attempts - 1:
                    wait_sec = (attempt + 1) * 2.0
                    logger.warning(
                        f"MWAPI {resp.status_code} server error (attempt {attempt + 1}/{total_attempts}), "
                        f"retrying in {wait_sec}s..."
                    )
                    time.sleep(wait_sec)
                    continue

                self._last_execution = ExecutionMetadata(
                    provider=self.provider_id,
                    provider_type=self.provider_type,
                    configured_model=model_name,
                    actual_model_used="",
                    status="FAILED",
                    error_type=err.error_type,
                    attempts=attempt + 1,
                    duration_seconds=round(time.time() - start_time, 3)
                )
                raise err

            except httpx.TimeoutException:
                if attempt < total_attempts - 1:
                    wait_sec = (attempt + 1) * 2.0
                    time.sleep(wait_sec)
                    continue

                self._last_execution = ExecutionMetadata(
                    provider=self.provider_id,
                    provider_type=self.provider_type,
                    configured_model=model_name,
                    actual_model_used="",
                    status="FAILED",
                    error_type="timeout",
                    attempts=attempt + 1,
                    duration_seconds=round(time.time() - start_time, 3)
                )
                raise AITimeoutError(
                    f"MWAPI request timed out after {timeout}s.",
                    provider=self.provider_id,
                    model=model_name
                )

            except (AIProviderError, AIAuthenticationError, AIQuotaExceededError,
                    AIRateLimitError, AITimeoutError, AINetworkError, AIInvalidResponseError):
                raise

            except Exception as e:
                clean_msg = sanitize_secrets(str(e), api_key)
                if attempt < total_attempts - 1:
                    time.sleep((attempt + 1) * 2.0)
                    continue

                self._last_execution = ExecutionMetadata(
                    provider=self.provider_id,
                    provider_type=self.provider_type,
                    configured_model=model_name,
                    actual_model_used="",
                    status="FAILED",
                    error_type="network",
                    attempts=attempt + 1,
                    duration_seconds=round(time.time() - start_time, 3)
                )
                raise AINetworkError(
                    f"MWAPI network connection error: {clean_msg}",
                    provider=self.provider_id,
                    model=model_name
                )

        raise AIServiceUnavailableError(
            f"MWAPI failed after {total_attempts} attempts.",
            provider=self.provider_id,
            model=model_name
        )

    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Validate MWAPI connection and API key by calling GET /v1/models.
        Consumes exactly ZERO generation tokens.
        """
        local_session = None
        target_db = db
        if target_db is None:
            try:
                from app.database import SessionLocal
                if SessionLocal:
                    local_session = SessionLocal()
                    target_db = local_session
            except Exception:
                pass

        try:
            api_key = get_mwapi_api_key(target_db)
            model_name = get_mwapi_model(target_db)
        finally:
            if local_session is not None:
                try:
                    local_session.close()
                except Exception:
                    pass

        if not api_key:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": False,
                "connected": False,
                "configured_model": model_name,
                "message": "MWAPI API key is not configured.",
                "error_type": "not_configured"
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.BASE_URL}/v1/models"

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)

            if resp.status_code == 200:
                return {
                    "provider": self.provider_id,
                    "provider_type": self.provider_type,
                    "configured": True,
                    "connected": True,
                    "configured_model": model_name,
                    "message": f"Kết nối MWAPI Gateway thành công. (Mô hình: {model_name})",
                    "error_type": None
                }

            err = classify_http_error(
                status_code=resp.status_code,
                response_text=resp.text,
                provider=self.provider_id,
                model=model_name,
                api_key=api_key
            )
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": True,
                "connected": False,
                "configured_model": model_name,
                "message": err.message,
                "error_type": err.error_type
            }

        except httpx.TimeoutException:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": True,
                "connected": False,
                "configured_model": model_name,
                "message": "MWAPI connection timed out after 15s.",
                "error_type": "timeout"
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": True,
                "connected": False,
                "configured_model": model_name,
                "message": sanitize_secrets(str(e), api_key),
                "error_type": "network"
            }

    def list_models(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """
        Dynamically list MWAPI models with 60-minute in-memory caching.
        Consumes ZERO generation tokens.
        """
        local_session = None
        target_db = db
        if target_db is None:
            try:
                from app.database import SessionLocal
                if SessionLocal:
                    local_session = SessionLocal()
                    target_db = local_session
            except Exception:
                pass

        try:
            api_key = get_mwapi_api_key(target_db)
        finally:
            if local_session is not None:
                try:
                    local_session.close()
                except Exception:
                    pass
        if not api_key:
            return []

        cached = global_model_cache.get(self.provider_id, api_key)
        if cached is not None:
            return cached

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.BASE_URL}/v1/models"

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)

            if resp.status_code != 200:
                logger.warning(f"Failed to discover MWAPI models: HTTP {resp.status_code}")
                return []

            data = resp.json()
            raw_list = data.get("data", [])
            if not isinstance(raw_list, list):
                return []

            models = []
            for item in raw_list:
                mid = str(item.get("id", "")).strip()
                if not mid:
                    continue
                display = str(item.get("name", mid)).strip()

                models.append({
                    "id": mid,
                    "name": display or mid,
                    "provider": self.provider_id,
                    "provider_type": self.provider_type,
                    "available": True
                })

            global_model_cache.set(self.provider_id, api_key, models)
            return models

        except Exception as e:
            logger.warning(f"Exception listing MWAPI models: {sanitize_secrets(str(e), api_key)}")
            return []

    def supports_model(self, model_name: str) -> bool:
        """
        Check if model is supported by MWAPI Gateway.
        As a multi-model gateway, accepts any non-empty model identifier.
        """
        if not model_name:
            return False
        return bool(model_name.strip())

    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """Return execution metadata from last operation without secrets."""
        if self._last_execution:
            return self._last_execution.to_dict()
        return ExecutionMetadata(provider=self.provider_id, provider_type=self.provider_type).to_dict()
