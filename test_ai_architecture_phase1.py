"""
Comprehensive Architecture & Contract Verification Test Suite for Multi-AI Phase 1

Verifies that:
1. GeminiProvider implements the provider contract (AIProvider).
2. AIProviderManager registers Gemini.
3. Default provider resolves to Gemini.
4. Manager delegates generation correctly.
5. Manager delegates test_connection correctly.
6. Manager delegates model listing correctly.
7. Business services (Script, Content, Research, AutoEditor, AudioSync) use AIProviderManager.
8. Existing Gemini retry behavior remains intact through the abstraction.
9. Existing 429 behavior remains intact through the abstraction.
10. Existing 503 behavior remains intact through the abstraction.
11. Execution metadata remains available and normalized.
12. API keys are never exposed in exceptions, metadata, or errors.
13. Unknown provider produces a clean controlled error.
14. No Ollama or local AI provider exists.
15. No fake OpenAI, Claude, OpenRouter, or Groq provider is registered yet.
16. Provider model support check functions accurately.
17. AIGenerationOptions passes options correctly.
18. Research low-consumption policy survives provider delegation.

All tests use mocked HTTP/services — ZERO real Gemini quota consumed.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import json
import httpx
from pathlib import Path

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
)
from app.services.ai.manager import (
    AIProviderManager,
    get_ai_manager,
    set_ai_manager,
)
from app.services.ai.providers.gemini import GeminiProvider
from app.services.gemini_service import (
    GeminiService,
    GeminiAPIError,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    _MODEL_CATALOG_CACHE,
    get_research_max_output_tokens,
)
from app.services.script_service import ScriptService
from app.services.content_service import ContentService
from app.services.auto_editor import AutoEditorService
from app.services.audio_sync import AudioSyncEngine


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


class TestMultiAIPhase1Architecture(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        _MODEL_CATALOG_CACHE["models"] = []
        _MODEL_CATALOG_CACHE["cached_at"] = 0.0
        # Reset singleton manager
        set_ai_manager(None)
        self._orig_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "AIzaSy_TEST_MOCK_KEY_FOR_PHASE1"

    def tearDown(self):
        set_ai_manager(None)
        if self._orig_key is not None:
            os.environ["GEMINI_API_KEY"] = self._orig_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)

    # 1. GeminiProvider implements the provider contract
    def test_01_gemini_provider_implements_contract(self):
        """GeminiProvider strictly implements AIProvider interface."""
        provider = GeminiProvider()
        self.assertIsInstance(provider, AIProvider)
        self.assertEqual(provider.provider_id, "gemini")
        self.assertEqual(provider.display_name, "Google Gemini")
        self.assertTrue(hasattr(provider, "generate"))
        self.assertTrue(hasattr(provider, "test_connection"))
        self.assertTrue(hasattr(provider, "list_models"))
        self.assertTrue(hasattr(provider, "supports_model"))
        self.assertTrue(hasattr(provider, "get_last_execution_metadata"))

    # 2. AIProviderManager registers Gemini
    def test_02_manager_registers_gemini(self):
        """AIProviderManager initializes with GeminiProvider registered."""
        mgr = AIProviderManager()
        providers = mgr.list_registered_providers()
        provider_ids = [p["provider_id"] for p in providers]
        self.assertIn("gemini", provider_ids)
        gemini_p = mgr.get_provider("gemini")
        self.assertIsInstance(gemini_p, GeminiProvider)

    # 3. Default provider resolves to Gemini
    def test_03_default_provider_resolves_to_gemini(self):
        """Default active provider resolves to Gemini without configuration changes."""
        mgr = AIProviderManager()
        active = mgr.get_active_provider()
        self.assertEqual(active.provider_id, "gemini")
        default_p = mgr.get_provider()
        self.assertEqual(default_p.provider_id, "gemini")

    # 4. Manager delegates generation correctly
    @patch("app.services.gemini_service.GeminiService.call_gemini", return_value="GENERATED_OUTPUT")
    def test_04_manager_delegates_generation(self, mock_call):
        """Manager.generate delegates to active Gemini provider."""
        mgr = AIProviderManager()
        result = mgr.generate("Hello world", db=self.mock_db, timeout=45.0, max_retries=1)
        self.assertEqual(result, "GENERATED_OUTPUT")
        mock_call.assert_called_once_with(
            prompt="Hello world",
            db=self.mock_db,
            timeout=45.0,
            max_retries=1,
            enable_fallback=True,
            max_output_tokens=None
        )

    # 5. Manager delegates test_connection correctly
    @patch("app.services.gemini_service.GeminiService.test_connection", return_value={
        "success": True, "message": "Connected", "model": "gemini-3.8-flash"
    })
    def test_05_manager_delegates_test_connection(self, mock_test):
        """Manager.test_connection delegates to GeminiProvider and attaches provider_id."""
        mgr = AIProviderManager()
        res = mgr.test_connection(db=self.mock_db)
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "gemini")
        self.assertEqual(res["model"], "gemini-3.8-flash")
        mock_test.assert_called_once_with(db=self.mock_db)

    # 6. Manager delegates model listing correctly
    @patch("app.services.ai.providers.gemini.get_api_key", return_value="AIzaSy_TEST_KEY")
    @patch("app.services.ai.providers.gemini.discover_available_models", return_value=["gemini-3.8-flash", "gemini-3.7-flash"])
    def test_06_manager_delegates_list_models(self, mock_disc, mock_key):
        """Manager.list_models delegates to active provider catalog discovery."""
        mgr = AIProviderManager()
        models = mgr.list_models(db=self.mock_db)
        model_ids = [m["id"] if isinstance(m, dict) else m for m in models]
        self.assertEqual(model_ids, ["gemini-3.8-flash", "gemini-3.7-flash"])
        mock_disc.assert_called_once_with("AIzaSy_TEST_KEY")

    # 7. Business services use AIProviderManager
    @patch("app.services.ai.manager.AIProviderManager.generate", return_value="Đây là kịch bản giới thiệu sản phẩm máy sấy thông minh tiện lợi cho mọi gia đình Việt Nam.")
    def test_07_script_service_uses_ai_manager(self, mock_gen):
        """ScriptService uses AIProviderManager.generate instead of direct GeminiService."""
        from app.models import Video, Product, Voice
        mock_video = MagicMock(spec=Video)
        mock_video.status = "NEW"
        mock_prod = MagicMock(spec=Product)
        mock_prod.name_vietnamese = "Máy sấy"
        mock_prod.content_angle = "Tiện lợi"
        mock_prod.hook = "Đừng bỏ qua"
        mock_video.product = mock_prod

        def query_side_effect(model):
            q = MagicMock()
            if model == Video:
                q.filter.return_value.first.return_value = mock_video
            elif model == Product:
                q.filter.return_value.first.return_value = mock_prod
            else:
                q.filter.return_value.first.return_value = None
            return q

        mock_db = MagicMock()
        mock_db.query.side_effect = query_side_effect

        svc = ScriptService()
        res = svc.generate_script_for_video(db=mock_db, video_id="V0001")
        self.assertTrue(res["success"])
        mock_gen.assert_called_once()

    @patch("app.services.ai.manager.AIProviderManager.generate")
    def test_07b_content_service_uses_ai_manager(self, mock_gen):
        """ContentService uses AIProviderManager.generate instead of direct GeminiService."""
        from app.models import Video, Voice, Product, Content
        valid_content = {
            "facebook_personal": {"caption": "Cap FB", "hashtags": "#test"},
            "facebook_page": {"caption": "Cap Page", "hashtags": "#test"},
            "tiktok": {"caption": "Cap TikTok", "hashtags": "#test"},
            "threads": {"caption": "Cap Threads", "hashtags": "#test"},
            "instagram": {"caption": "Cap IG", "hashtags": "#test"},
            "shopee": {"caption": "Cap Shopee", "hashtags": "#test"},
            "youtube": {"title": "Title YT", "description": "Desc YT", "hashtags": "#test"}
        }
        mock_gen.return_value = json.dumps(valid_content)

        mock_video = MagicMock(spec=Video)
        mock_video.status = "SCRIPT_READY"
        mock_voice = MagicMock(spec=Voice)
        mock_voice.script = "Kịch bản sản phẩm"
        mock_prod = MagicMock(spec=Product)
        mock_prod.name_vietnamese = "Bình nước"
        mock_video.product = mock_prod

        def query_side_effect(model):
            q = MagicMock()
            if model == Video:
                q.filter.return_value.first.return_value = mock_video
            elif model == Voice:
                q.filter.return_value.first.return_value = mock_voice
            elif model == Product:
                q.filter.return_value.first.return_value = mock_prod
            else:
                q.filter.return_value.first.return_value = None
            return q

        mock_db = MagicMock()
        mock_db.query.side_effect = query_side_effect

        svc = ContentService()
        res = svc.generate_content_for_video(db=mock_db, video_id="V0001")
        self.assertTrue(res["success"])
        mock_gen.assert_called_once()

    # 8. Existing Gemini retry behavior remains intact through the abstraction
    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_success")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_08_gemini_retry_behavior_intact(self, mock_status, mock_fail, mock_succ, mock_start, mock_m, mock_k, mock_sleep):
        """503 temporary overload retries and recovers through AIProviderManager."""
        resp_503 = _make_resp(503, {"error": {"message": "High demand"}})
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": "RECOVERED_VIA_MANAGER"}]}}]})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.side_effect = [resp_503, resp_200]
            mock_client_cls.return_value = mock_client

            mgr = get_ai_manager()
            result = mgr.generate("test prompt", db=self.mock_db, max_retries=2, enable_fallback=False)
            self.assertEqual(result, "RECOVERED_VIA_MANAGER")
            self.assertEqual(mock_client.post.call_count, 2)
            self.assertEqual(mock_sleep.call_count, 1)

    # 9. Existing 429 behavior remains intact through the abstraction
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_09_gemini_429_quota_fast_fail(self, mock_status, mock_fail, mock_start, mock_m, mock_k):
        """HTTP 429 daily quota raises AIQuotaExceededError with 0 retries."""
        resp_429 = _make_resp(429, {"error": {"message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = resp_429
            mock_client_cls.return_value = mock_client

            mgr = get_ai_manager()
            with self.assertRaises(AIQuotaExceededError) as ctx:
                mgr.generate("test prompt", db=self.mock_db, max_retries=2)

            self.assertIsInstance(ctx.exception, GeminiQuotaExceededError)
            self.assertIsInstance(ctx.exception, AIQuotaExceededError)
            self.assertEqual(mock_client.post.call_count, 1)

    # 10. Existing 503 behavior remains intact through the abstraction
    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_success")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_failure")
    @patch("app.services.gemini_status.GeminiStatusTracker.update_status")
    def test_10_gemini_503_fallback_intact(self, mock_status, mock_fail, mock_succ, mock_start, mock_m, mock_k, mock_sleep):
        """Primary 503 triggers smart fallback to available model through AIProviderManager."""
        resp_503 = _make_resp(503, {"error": {"message": "High demand"}})
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": "FALLBACK_VIA_MANAGER"}]}}]})

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

            mgr = get_ai_manager()
            result = mgr.generate("test prompt", db=self.mock_db, max_retries=2, enable_fallback=True)
            self.assertEqual(result, "FALLBACK_VIA_MANAGER")

            meta = mgr.get_last_execution_metadata()
            self.assertTrue(meta["fallback_used"])
            self.assertEqual(meta["configured_model"], "gemini-3.8-flash")
            self.assertEqual(meta["actual_model_used"], "gemini-3.7-flash")

    # 11. Execution metadata remains available and normalized
    def test_11_execution_metadata_structure(self):
        """ExecutionMetadata produces secret-free, normalized audit dictionary."""
        meta = ExecutionMetadata(
            provider="gemini",
            configured_model="gemini-3.8-flash",
            actual_model_used="gemini-3.7-flash",
            fallback_used=True,
            status="SUCCESS",
            error_type="503",
            attempts=3
        )
        d = meta.to_dict()
        self.assertEqual(d["provider"], "gemini")
        self.assertEqual(d["configured_model"], "gemini-3.8-flash")
        self.assertEqual(d["actual_model_used"], "gemini-3.7-flash")
        self.assertTrue(d["fallback_used"])
        self.assertEqual(d["status"], "SUCCESS")
        self.assertEqual(d["error_type"], "503")
        self.assertEqual(d["attempts"], 3)
        # Ensure no api_key or auth key exists in metadata dict
        self.assertNotIn("api_key", d)
        self.assertNotIn("secret", d)
        self.assertNotIn("key", d)

    # 12. API keys are never exposed in exceptions, metadata, or errors
    def test_12_api_keys_never_exposed(self):
        """API key is redacted if an error message contains raw authentication info."""
        from app.services.gemini_service import sanitize_error_message
        secret = "AIzaSy_SUPER_CONFIDENTIAL_KEY_9999"
        raw_msg = f"Error communicating with Google: key={secret} header x-goog-api-key: {secret}"
        sanitized = sanitize_error_message(raw_msg, secret)
        self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED]", sanitized)

    # 13. Unknown provider produces a clean controlled error
    def test_13_unknown_provider_produces_clean_error(self):
        """Requesting an unregistered provider raises controlled AIProviderError."""
        mgr = AIProviderManager()
        with self.assertRaises(AIProviderError) as ctx:
            mgr.get_provider("nonexistent_provider")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.provider, "nonexistent_provider")
        self.assertIn("nonexistent_provider", str(ctx.exception))

    # 14. No Ollama or local AI provider exists
    def test_14_no_ollama_or_local_provider(self):
        """Strictly enforce that no Ollama or local AI provider is introduced in Phase 1."""
        mgr = AIProviderManager()
        providers = mgr.list_registered_providers()
        provider_ids = [p["provider_id"].lower() for p in providers]
        self.assertNotIn("ollama", provider_ids)
        self.assertNotIn("local", provider_ids)
        self.assertNotIn("llama", provider_ids)

    # 15. Exactly 5 official cloud providers are registered in Phase 2
    def test_15_registered_providers_phase2(self):
        """Phase 2 registers exactly 5 real cloud providers; unknown providers raise AIProviderError."""
        mgr = AIProviderManager()
        providers = mgr.list_registered_providers()
        provider_ids = [p["provider_id"] for p in providers]
        self.assertEqual(len(providers), 6, "Exactly 6 providers must be registered (including mwapi).")
        self.assertEqual(sorted(provider_ids), ["anthropic", "gemini", "groq", "mwapi", "openai", "openrouter"])

        for pid in ["gemini", "openai", "anthropic", "groq", "openrouter", "mwapi"]:
            p = mgr.get_provider(pid)
            self.assertIsNotNone(p)
            self.assertEqual(p.provider_id, pid)

        for unreg in ["ollama", "local", "fake_provider", "unknown"]:
            with self.assertRaises(AIProviderError):
                mgr.get_provider(unreg)

    # 16. Provider model support check functions accurately
    def test_16_supports_model_check(self):
        """GeminiProvider accurately checks model support."""
        provider = GeminiProvider()
        self.assertTrue(provider.supports_model("gemini-3.8-flash"))
        self.assertTrue(provider.supports_model("gemini-3.7-flash"))
        self.assertTrue(provider.supports_model("gemini-3.5-flash-lite"))
        self.assertFalse(provider.supports_model("gpt-4o"))
        self.assertFalse(provider.supports_model("claude-3-5-sonnet"))
        self.assertFalse(provider.supports_model("llama-3"))
        self.assertFalse(provider.supports_model(""))

    # 17. AIGenerationOptions passes options correctly
    def test_17_options_dataclass(self):
        """AIGenerationOptions sets and forwards options properly."""
        opts = AIGenerationOptions(
            timeout=30.0,
            max_retries=1,
            enable_fallback=False,
            temperature=0.7,
            max_output_tokens=1000
        )
        self.assertEqual(opts.timeout, 30.0)
        self.assertEqual(opts.max_retries, 1)
        self.assertFalse(opts.enable_fallback)
        self.assertEqual(opts.temperature, 0.7)
        self.assertEqual(opts.max_output_tokens, 1000)

    # 18. Research policy survives provider delegation
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_18_research_policy_survives_provider_delegation(self, mock_post, mock_model, mock_key):
        """AIProviderManager.generate_products maintains max_retries=0 and bounded max_output_tokens."""
        sample_json = json.dumps([{"name_vietnamese": "Sản phẩm A"}])
        resp_200 = _make_resp(200, {"candidates": [{"content": {"parts": [{"text": sample_json}]}}]})
        mock_post.return_value = resp_200

        mgr = get_ai_manager()
        prods = mgr.generate_products(niche="Gia dụng", count=10, db=self.mock_db)

        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1)
        # Verify bounded output token budget was sent
        called_json = mock_post.call_args[1]["json"]
        self.assertEqual(called_json["generationConfig"]["maxOutputTokens"], 1500)


if __name__ == "__main__":
    unittest.main()
