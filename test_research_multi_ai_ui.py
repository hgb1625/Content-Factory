"""
Unit & Integration Tests for Research UI Multi-AI Display & Invariants.

Verifies:
1. Dynamic provider & model rendering for Gemini (e.g. Gemini, gemini-2.5-flash).
2. Dynamic provider & model rendering for OpenAI (e.g. OpenAI, gpt-4o-mini).
3. Quota / rate-limit / overload / auth error messages identify actual provider (e.g. "OpenAI đã hết hạn mức API / quota").
4. Gemini diagnostics widgets do not masquerade as generic Research provider status.
5. Strict Research invariant:
   - cache hit = 0 generation calls
   - cache miss = exactly 1 generation call, 0 retries, 0 fallback
6. Zero network calls for UI rendering.
7. Zero API key exposure.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from starlette.testclient import TestClient

from app.database import get_db
from app.main import app
from app.models import Product, Setting
from app.services.ai import (
    AIAuthenticationError,
    AINetworkError,
    AIProviderError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    get_ai_manager,
)
from app.services.gemini_service import GeminiQuotaExceededError


SAMPLE_PRODUCTS = [
    {
        "name_vietnamese": "Nồi cơm điện mini đa năng",
        "name_chinese": "多功能迷你电饭煲",
        "douyin_keywords": "多功能电饭煲 独居好物 宿舍神器",
        "content_angle": "Tiện lợi cho người sống một mình",
        "hook": "Bữa cơm ngon chỉ mất 15 phút với chiếc nồi này!",
    }
]


def _make_mock_db(active_provider="gemini", models=None, keys=None, products=None):
    """Create in-memory SQLite mock session for settings and products."""
    models = models or {}
    keys = keys or {}
    products = products or []

    settings_store = {
        "active_ai_provider": active_provider,
        "gemini_model": models.get("gemini", "gemini-2.5-flash"),
        "openai_model": models.get("openai", "gpt-4o-mini"),
        "anthropic_model": models.get("anthropic", "claude-3-5-haiku-20241022"),
        "groq_model": models.get("groq", "llama-3.3-70b-versatile"),
        "gemini_api_key": keys.get("gemini", "AIzaSy_TEST_KEY_GEMINI_12345"),
        "openai_api_key": keys.get("openai", "sk-proj-TEST_KEY_OPENAI_12345"),
        "anthropic_api_key": keys.get("anthropic", "sk-ant-TEST_KEY_ANTHROPIC_12345"),
        "groq_api_key": keys.get("groq", "gsk_TEST_KEY_GROQ_12345"),
    }

    mock_session = MagicMock()

    def _query_side_effect(model_cls):
        q = MagicMock()
        if model_cls is Setting:
            def _filter_side_effect(*args, **kwargs):
                f = MagicMock()
                def _first_side_effect():
                    try:
                        expr = args[0]
                        k = getattr(getattr(expr, "right", None), "value", None)
                        if isinstance(k, (list, tuple, set)):
                            for cand in k:
                                if cand in settings_store:
                                    s = MagicMock()
                                    s.key = cand
                                    s.value = settings_store[cand]
                                    return s
                        elif isinstance(k, str) and k in settings_store:
                            s = MagicMock()
                            s.key = k
                            s.value = settings_store[k]
                            return s
                    except Exception:
                        pass
                    return None
                f.first = _first_side_effect
                f.all = lambda: []
                return f
            q.filter = _filter_side_effect
            q.all = lambda: []
            return q
        elif model_cls is Product:
            q.all = lambda: list(products)
            q.order_by.return_value.all = lambda: list(products)
            return q
        return q

    mock_session.query.side_effect = _query_side_effect
    return mock_session, settings_store


class TestResearchMultiAIUI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)

    # -------------------------------------------------------------------------
    # 1. Active Provider = Gemini Dynamic Display
    # -------------------------------------------------------------------------
    def test_01_research_page_renders_gemini_dynamically(self):
        db, _ = _make_mock_db(
            active_provider="gemini",
            models={"gemini": "gemini-2.5-flash"},
            keys={"gemini": "AIzaSy_TEST_KEY_GEMINI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # 1. Header has Gemini
        self.assertIn("Tạo Danh Sách Sản Phẩm Tiềm Năng (Gemini)", html)
        # 2. Model badge has gemini-2.5-flash
        self.assertIn("gemini-2.5-flash", html)
        # 3. Provider name is Gemini
        self.assertIn('id="researchProviderName">Gemini</span>', html)
        # 4. Info card reflects Gemini
        self.assertIn("<strong class=\"text-white\">Gemini</strong> tự động phân tích", html)
        # 5. Client submission script specifies active provider
        self.assertIn('const activeProviderName = "Gemini";', html)
        self.assertIn('const activeProviderId = "gemini";', html)

    # -------------------------------------------------------------------------
    # 2. Active Provider = OpenAI Dynamic Display
    # -------------------------------------------------------------------------
    def test_02_research_page_renders_openai_dynamically(self):
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # 1. Header has OpenAI
        self.assertIn("Tạo Danh Sách Sản Phẩm Tiềm Năng (OpenAI)", html)
        # 2. Model badge has gpt-4o-mini
        self.assertIn("gpt-4o-mini", html)
        # 3. Provider name is OpenAI
        self.assertIn('id="researchProviderName">OpenAI</span>', html)
        # 4. Status message says OpenAI is configured and ready
        self.assertIn("OpenAI đã cấu hình và sẵn sàng", html)
        # 5. Stale generic text must NOT appear in provider status box
        self.assertNotIn('gemini-3.8-flash</span>', html)
        # 6. Info card reflects OpenAI
        self.assertIn("<strong class=\"text-white\">OpenAI</strong> tự động phân tích", html)
        # 7. Client submission script specifies active provider
        self.assertIn('const activeProviderName = "OpenAI";', html)
        self.assertIn('const activeProviderId = "openai";', html)

    # -------------------------------------------------------------------------
    # 3. Quota Exceeded Error on OpenAI displays "OpenAI đã hết hạn mức API / quota"
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_03_openai_quota_exceeded_error_display(self, mock_get_manager):
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("openai")
        mock_mgr.generate_products.side_effect = AIQuotaExceededError(
            "You have exceeded your current quota, please check your plan and billing details.",
            provider="openai",
            model="gpt-4o-mini"
        )
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post("/research", data={"niche": "Gia dụng thông minh", "product_count": 10, "fresh": "true"})
        self.assertEqual(resp.status_code, 429)
        html = resp.text

        # 1. Requirement 4: "OpenAI đã hết hạn mức API / quota"
        self.assertIn("OpenAI đã hết hạn mức API / quota", html)
        # 2. Context error identifies OpenAI
        self.assertIn("Đã chạm giới hạn hạn mức OpenAI (HTTP 429)", html)
        # 3. No stale Gemini error
        self.assertNotIn("Đã hết hạn mức Gemini hôm nay", html)
        self.assertNotIn("giới hạn hạn mức Gemini", html)

    # -------------------------------------------------------------------------
    # 4. Quota Exceeded Error on Gemini displays "Gemini đã hết hạn mức API / quota"
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_04_gemini_quota_exceeded_error_display(self, mock_get_manager):
        db, _ = _make_mock_db(
            active_provider="gemini",
            models={"gemini": "gemini-2.5-flash"},
            keys={"gemini": "AIzaSy_TEST_KEY_GEMINI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("gemini")
        mock_mgr.generate_products.side_effect = GeminiQuotaExceededError("Resource has been exhausted (e.g. check quota).")
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post("/research", data={"niche": "Đồ chơi công nghệ", "product_count": 5, "fresh": "true"})
        self.assertEqual(resp.status_code, 429)
        html = resp.text

        self.assertIn("Gemini đã hết hạn mức API / quota", html)
        self.assertIn("Đã chạm giới hạn hạn mức Gemini (HTTP 429)", html)

    # -------------------------------------------------------------------------
    # 5. Service Unavailable (503) identifies active provider
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_05_openai_service_unavailable_error_display(self, mock_get_manager):
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("openai")
        mock_mgr.generate_products.side_effect = AIServiceUnavailableError(
            "The server is temporarily unavailable or overloaded.",
            provider="openai",
            model="gpt-4o-mini"
        )
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post("/research", data={"niche": "Gia dụng", "product_count": 5, "fresh": "true"})
        self.assertEqual(resp.status_code, 503)
        html = resp.text

        self.assertIn("OpenAI đang quá tải", html)
        self.assertIn("OpenAI đang tạm thời quá tải (HTTP 503)", html)
        self.assertNotIn("Gemini đang quá tải", html)

    # -------------------------------------------------------------------------
    # 6. Authentication Error (401) identifies active provider
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_06_openai_auth_error_display(self, mock_get_manager):
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("openai")
        mock_mgr.generate_products.side_effect = AIAuthenticationError(
            "Incorrect API key provided.",
            provider="openai"
        )
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post("/research", data={"niche": "Gia dụng", "product_count": 5, "fresh": "true"})
        self.assertEqual(resp.status_code, 400)
        html = resp.text

        self.assertIn("Lỗi xác thực OpenAI", html)
        self.assertIn("Lỗi OpenAI API Key (HTTP 401)", html)
        self.assertNotIn("Lỗi xác thực Gemini", html)

    # -------------------------------------------------------------------------
    # 7. Timeout and Network Errors identify active provider
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_07_openai_timeout_and_network_error_display(self, mock_get_manager):
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("openai")
        mock_mgr.generate_products.side_effect = AITimeoutError("OpenAI request timed out", provider="openai")
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post("/research", data={"niche": "Gia dụng", "product_count": 5, "fresh": "true"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("OpenAI hết thời gian chờ", resp.text)
        self.assertIn("Yêu cầu tới OpenAI bị quá thời gian chờ", resp.text)

    # -------------------------------------------------------------------------
    # 8. Research Invariant: Cache Hit = 0 AI calls
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_08_research_cache_hit_zero_ai_calls(self, mock_get_manager):
        cached_product = Product(
            product_id="P0001",
            niche="Đồ gia dụng",
            name_vietnamese="Chảo chống dính",
            name_chinese="不粘锅",
            douyin_keywords="不粘锅推荐",
            content_angle="Nấu ăn",
            hook="Chảo xịn",
            status="RESEARCHED"
        )
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"},
            products=[cached_product]
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_get_manager.return_value = mock_mgr

        # Request 1 product for matching niche without fresh flag
        resp = self.client.post(
            "/research",
            data={"niche": "Đồ gia dụng", "product_count": 1, "fresh": "false"},
            follow_redirects=False
        )

        # 303 Redirect to /products?cached=1
        self.assertEqual(resp.status_code, 303)
        self.assertIn("cached=1", resp.headers["location"])
        # Exactly 0 calls to AI manager
        mock_mgr.generate_products.assert_not_called()

    # -------------------------------------------------------------------------
    # 9. Research Invariant: Cache Miss = Exactly 1 AI call, no fallback/retries
    # -------------------------------------------------------------------------
    @patch("app.routes.research.get_ai_manager")
    def test_09_research_cache_miss_exactly_one_ai_call(self, mock_get_manager):
        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"},
            products=[]
        )
        app.dependency_overrides[get_db] = lambda: db

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("openai")
        mock_mgr.generate_products.return_value = [
            {**SAMPLE_PRODUCTS[0], "name_vietnamese": f"Sản phẩm gia dụng {i+1}"}
            for i in range(5)
        ]
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post(
            "/research",
            data={"niche": "Gia dụng mới", "product_count": 5, "fresh": "true"},
            follow_redirects=False
        )

        # 303 Redirect to /products
        self.assertEqual(resp.status_code, 303)
        # Exactly 1 call to generate_products
        self.assertEqual(mock_mgr.generate_products.call_count, 1)

    # -------------------------------------------------------------------------
    # 10. Zero Network Calls for UI Rendering
    # -------------------------------------------------------------------------
    @patch("app.services.ai.providers.openai.httpx.Client")
    @patch("app.services.gemini_service.httpx.Client")
    def test_10_zero_network_calls_during_ui_rendering(self, mock_gemini_client, mock_openai_client):
        mock_gemini_client.side_effect = RuntimeError("NETWORK CALL FORBIDDEN IN UI RENDERING")
        mock_openai_client.side_effect = RuntimeError("NETWORK CALL FORBIDDEN IN UI RENDERING")

        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": "sk-proj-TEST_KEY_OPENAI_12345"}
        )
        app.dependency_overrides[get_db] = lambda: db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_gemini_client.call_count, 0)
        self.assertEqual(mock_openai_client.call_count, 0)

    # -------------------------------------------------------------------------
    # 11. Zero API Key Exposure
    # -------------------------------------------------------------------------
    def test_11_zero_api_key_exposure(self):
        openai_secret = "sk-proj-SECRET_OPENAI_KEY_ABCDEF123456789"
        gemini_secret = "AIzaSy_SECRET_GEMINI_KEY_ABCDEF123456789"

        db, _ = _make_mock_db(
            active_provider="openai",
            models={"openai": "gpt-4o-mini"},
            keys={"openai": openai_secret, "gemini": gemini_secret}
        )
        app.dependency_overrides[get_db] = lambda: db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(openai_secret, resp.text)
        self.assertNotIn(gemini_secret, resp.text)


if __name__ == "__main__":
    unittest.main()
