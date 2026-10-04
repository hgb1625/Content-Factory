"""
Comprehensive Unit Test Suite for Gemini Reliability & Smart Fallback
All tests use mocked HTTP responses — zero real Gemini quota consumed.
"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import httpx

from app.services.gemini_service import (
    GeminiService,
    GeminiAPIError,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    GeminiBadRequestError,
    GeminiAuthError,
    GeminiPermissionError,
    GeminiModelNotFoundError,
    GeminiServiceUnavailableError,
    GeminiTimeoutError,
    GeminiNetworkError,
    sanitize_error_message,
    select_fallback_model,
    discover_available_models,
    _MODEL_CATALOG_CACHE
)


def _make_resp(status_code: int, data: dict = None, text: str = ""):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {}
    if data is not None:
        resp.json.return_value = data
        resp.text = json.dumps(data)
    else:
        resp.json.side_effect = Exception("Not JSON")
        resp.text = text
    return resp


class TestGeminiReliability(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        _MODEL_CATALOG_CACHE["models"] = []
        _MODEL_CATALOG_CACHE["cached_at"] = 0.0

    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_success")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_01_503_then_success(self, mock_status, mock_fail, mock_succ, mock_start, mock_m, mock_k, mock_sleep):
        """HTTP 503 on first attempt followed by HTTP 200 succeeds on retry."""
        resp_503 = _make_resp(503, {"error": {"message": "High demand"}})
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": "RECOVERED_OK"}]}}]})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.side_effect = [resp_503, resp_200]
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            result = svc.call_gemini("test prompt", db=self.mock_db, max_retries=2)
            self.assertEqual(result, "RECOVERED_OK")
            self.assertEqual(mock_client.post.call_count, 2)
            self.assertEqual(mock_sleep.call_count, 1)

    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_02_repeated_503_exhaustion_without_fallback(self, mock_status, mock_fail, mock_start, mock_m, mock_k, mock_sleep):
        """Repeated 503 errors exhaust retries and raise GeminiServiceUnavailableError."""
        resp_503 = _make_resp(503, {"error": {"message": "High demand continuously"}})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_503
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            with self.assertRaises(GeminiServiceUnavailableError) as ctx:
                svc.call_gemini("test prompt", db=self.mock_db, max_retries=2, enable_fallback=False)

            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("High demand continuously", ctx.exception.upstream_message)
            self.assertEqual(mock_client.post.call_count, 3)

    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_success")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_03_503_primary_plus_fallback_success(self, mock_status, mock_fail, mock_succ, mock_start, mock_m, mock_k, mock_sleep):
        """When primary model gemini-3.8-flash returns 503, fallback to available model succeeds."""
        resp_503 = _make_resp(503, {"error": {"message": "This model is currently experiencing high demand."}})
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": "FALLBACK_OK"}]}}]})

        def mock_post(url, *args, **kwargs):
            if "gemini-3.8-flash" in url:
                return resp_503
            return resp_200

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.side_effect = mock_post
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            result = svc.call_gemini("test prompt", db=self.mock_db, max_retries=2, enable_fallback=True)
            self.assertEqual(result, "FALLBACK_OK")
            self.assertTrue(svc.last_execution["fallback_used"])
            self.assertEqual(svc.last_execution["primary_model"], "gemini-3.8-flash")
            self.assertEqual(svc.last_execution["actual_model_used"], "gemini-3.7-flash")
            self.assertEqual(svc.last_execution["primary_failure"], 503)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_04_401_auth_error_no_fallback(self, mock_status, mock_fail, mock_start, mock_m, mock_k):
        """HTTP 401 raises GeminiAuthError immediately without retrying or fallback."""
        resp_401 = _make_resp(401, {"error": {"message": "API key not valid."}})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_401
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            with self.assertRaises(GeminiAuthError) as ctx:
                svc.call_gemini("test prompt", db=self.mock_db, max_retries=2, enable_fallback=True)

            self.assertEqual(ctx.exception.status_code, 401)
            self.assertEqual(mock_client.post.call_count, 1)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_05_403_permission_error_no_fallback(self, mock_status, mock_fail, mock_start, mock_m, mock_k):
        """HTTP 403 raises GeminiPermissionError without retrying or fallback."""
        resp_403 = _make_resp(403, {"error": {"message": "Permission denied for resource."}})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_403
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            with self.assertRaises(GeminiPermissionError) as ctx:
                svc.call_gemini("test prompt", db=self.mock_db, max_retries=2, enable_fallback=True)

            self.assertEqual(ctx.exception.status_code, 403)
            self.assertEqual(mock_client.post.call_count, 1)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-nonexistent-model")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_06_404_model_not_found_no_fallback(self, mock_status, mock_fail, mock_start, mock_m, mock_k):
        """HTTP 404 raises GeminiModelNotFoundError without fallback."""
        resp_404 = _make_resp(404, {"error": {"message": "Model not found."}})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_404
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            with self.assertRaises(GeminiModelNotFoundError) as ctx:
                svc.call_gemini("test prompt", db=self.mock_db, max_retries=2, enable_fallback=True)

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertEqual(mock_client.post.call_count, 1)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_07_429_daily_quota_fast_fail(self, mock_status, mock_fail, mock_start, mock_m, mock_k):
        """HTTP 429 Daily Quota fails fast with 0 retries and no fallback."""
        resp_429 = _make_resp(429, {"error": {"message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_429
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            with self.assertRaises(GeminiQuotaExceededError) as ctx:
                svc.call_gemini("test prompt", db=self.mock_db, max_retries=2, enable_fallback=True)

            self.assertEqual(ctx.exception.status_code, 429)
            self.assertEqual(mock_client.post.call_count, 1)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_08_network_timeout(self, mock_status, mock_fail, mock_start, mock_m, mock_k):
        """TimeoutException maps to GeminiTimeoutError."""
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.side_effect = httpx.TimeoutException("Timed out")
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            with self.assertRaises(GeminiTimeoutError):
                svc.call_gemini("test prompt", db=self.mock_db, max_retries=0)

    def test_09_upstream_error_sanitization_and_api_key_protection(self):
        """Verify API key is completely sanitized from messages, URLs, and headers."""
        secret_key = "AIzaSy_SECRET_KEY_1234567890abcdef"
        raw_error = f"Failed to call https://generativelanguage.googleapis.com/v1beta/models?key={secret_key} with header x-goog-api-key: {secret_key}"
        clean = sanitize_error_message(raw_error, secret_key)
        self.assertNotIn(secret_key, clean)
        self.assertIn("[REDACTED]", clean)

    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_success")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_10_test_connection_503_fallback_reporting(self, mock_status, mock_fail, mock_succ, mock_start, mock_m, mock_k, mock_sleep):
        """test_connection returns transparent fallback metadata when primary experiences 503."""
        resp_503 = _make_resp(503, {"error": {"message": "High demand spike"}})
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})

        def mock_post(url, *args, **kwargs):
            if "gemini-3.8-flash" in url:
                return resp_503
            return resp_200

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.side_effect = mock_post
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            report = svc.test_connection(db=self.mock_db)
            self.assertTrue(report["success"])
            self.assertTrue(report["fallback_used"])
            self.assertIn("gemini-3.8-flash", report["message"])
            self.assertIn("HTTP 503", report["message"])
            self.assertEqual(report["model"], "gemini-3.8-flash")
            self.assertIn("actual_model", report)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_success")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_11_model_discovery_only_when_needed(self, mock_status, mock_succ, mock_start, mock_m, mock_k):
        """Model discovery is NOT invoked when the primary model call succeeds."""
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": "PRIMARY_SUCCESS"}]}}]})

        with patch("httpx.Client") as mock_client_cls, \
             patch("app.services.gemini_service.discover_available_models") as mock_discover:

            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_200
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            res = svc.call_gemini("test prompt", db=self.mock_db, enable_fallback=True)
            self.assertEqual(res, "PRIMARY_SUCCESS")
            # Model catalog discovery must NOT be called when primary succeeds
            mock_discover.assert_not_called()

    def test_12_start_bat_recognizes_venv(self):
        """START.bat includes .venv\\Scripts\\python.exe in launcher sequence."""
        start_bat = Path(__file__).resolve().parent / "START.bat"
        self.assertTrue(start_bat.exists(), "START.bat must exist")
        content = start_bat.read_text(encoding="utf-8")
        self.assertIn(".venv\\Scripts\\python.exe", content)
        # Check order: .venv_vieneu_new or .venv_vieneu before .venv
        pos_vieneu = max(content.find(".venv_vieneu_new"), content.find(".venv_vieneu"))
        pos_venv = content.find(".venv\\Scripts\\python.exe")
        pos_python = content.find("where python")
        self.assertTrue(pos_vieneu < pos_venv < pos_python, "Priority order must be VieNeu -> .venv -> PATH")


if __name__ == "__main__":
    unittest.main()
