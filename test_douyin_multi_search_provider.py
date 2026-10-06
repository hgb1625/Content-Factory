"""
Unit & Integration Tests for Douyin Multi-Search Provider Architecture (24-Point Test Suite)
All tests run OFFLINE using mocks. ZERO external API calls.
"""

import sys
import unittest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal, init_db
from app.models import Video
from app.services.douyin_search import (
    DouyinSearchProvider,
    DouyinSearchResult,
    SerpApiSearchProvider,
    BrowserSearchProvider,
    DouyinSearchManager,
    douyin_search_manager,
)
from app.services.douyin_search.models import (
    extract_canonical_douyin_video_id,
    deduplicate_against_db,
)
from app.services.downloader_service import DownloaderService


class TestDouyinMultiSearchProvider(unittest.TestCase):

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.client = TestClient(app)
        self.serpapi_provider = SerpApiSearchProvider(api_key="mock_test_key_12345", cache_ttl=1200)
        self.serpapi_provider.clear_cache()

    def tearDown(self):
        self.db.close()
        self.serpapi_provider.clear_cache()

    # 1. Provider interface
    def test_01_provider_interface(self):
        self.assertTrue(issubclass(SerpApiSearchProvider, DouyinSearchProvider))
        self.assertTrue(issubclass(BrowserSearchProvider, DouyinSearchProvider))
        p = self.serpapi_provider
        self.assertEqual(p.provider_id, "serpapi")
        self.assertEqual(p.display_name, "No Login — Google Index")
        self.assertFalse(p.requires_douyin_login)
        self.assertTrue(hasattr(p, "search"))

    # 2. Provider selection
    def test_02_provider_selection(self):
        manager = DouyinSearchManager()
        p_serpapi = manager.get_provider("serpapi")
        self.assertIsInstance(p_serpapi, SerpApiSearchProvider)

        p_browser = manager.get_provider("browser")
        self.assertIsInstance(p_browser, BrowserSearchProvider)

        # Default fallback to serpapi on unknown name
        p_default = manager.get_provider("unknown_provider")
        self.assertIsInstance(p_default, SerpApiSearchProvider)

        providers_list = manager.list_providers()
        self.assertEqual(len(providers_list), 2)
        ids = [p["id"] for p in providers_list]
        self.assertIn("serpapi", ids)
        self.assertIn("browser", ids)

    # 3. Query construction
    def test_03_query_construction(self):
        captured_params = {}
        def mock_get(url, params=None, **kwargs):
            nonlocal captured_params
            captured_params = params or {}
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"organic_results": []}
            return m

        with patch("httpx.Client.get", side_effect=mock_get):
            self.serpapi_provider.search("宿舍迷你电饭煲")
            self.assertIn("q", captured_params)
            self.assertEqual(captured_params["q"], "site:douyin.com/video 宿舍迷你电饭煲")
            self.assertEqual(captured_params["engine"], "google")

    # 4. Chinese keyword encoding
    def test_04_chinese_keyword_encoding(self):
        captured_params = {}
        def mock_get(url, params=None, **kwargs):
            nonlocal captured_params
            captured_params = params or {}
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"organic_results": []}
            return m

        kw = "便携式电热杯"
        with patch("httpx.Client.get", side_effect=mock_get):
            self.serpapi_provider.search(kw)
            self.assertIn(kw, captured_params["q"])
            self.assertEqual(captured_params["hl"], "zh-cn")

    # 5. Canonical video ID extraction
    def test_05_canonical_video_id_extraction(self):
        vid1 = extract_canonical_douyin_video_id("https://www.douyin.com/video/7461911073162791080")
        self.assertEqual(vid1, "7461911073162791080")
        vid2 = extract_canonical_douyin_video_id("https://www.douyin.com/jingxuan/search/abc?modal_id=7461911073162791099")
        self.assertEqual(vid2, "7461911073162791099")

    # 6. Reject non-Douyin URLs
    def test_06_reject_non_douyin_urls(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.youtube.com/watch?v=12345", "title": "YouTube Video"},
                {"link": "https://www.bilibili.com/video/BV1xx411c7mD", "title": "Bilibili Video"},
                {"link": "https://www.douyin.com/video/7461911073162791001", "title": "Valid Douyin Video"}
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("test")
            self.assertTrue(res["success"])
            self.assertEqual(len(res["videos"]), 1)
            self.assertEqual(res["videos"][0]["video_id"], "7461911073162791001")

    # 7. Reject non-video Douyin pages
    def test_07_reject_non_video_douyin_pages(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.douyin.com/user/MS4wLjABAAAA...", "title": "User Profile"},
                {"link": "https://www.douyin.com/search/cooking", "title": "Search Page"},
                {"link": "https://www.douyin.com/jingxuan", "title": "Jingxuan Feed"},
                {"link": "https://live.douyin.com/12345678", "title": "Live Stream"},
                {"link": "https://www.douyin.com/video/7461911073162791002", "title": "Valid Video"}
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("test")
            self.assertEqual(len(res["videos"]), 1)
            self.assertEqual(res["videos"][0]["video_id"], "7461911073162791002")

    # 8. Duplicate URLs
    def test_08_duplicate_urls(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.douyin.com/video/7461911073162791001", "title": "Vid A"},
                {"link": "https://www.douyin.com/video/7461911073162791001", "title": "Vid A Duplicate"},
                {"link": "https://www.douyin.com/video/7461911073162791002", "title": "Vid B"}
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("test")
            self.assertEqual(len(res["videos"]), 2)

    # 9. DB duplicate detection
    def test_09_db_duplicate_detection(self):
        test_url = "https://www.douyin.com/video/7461911073162791999"
        self.db.query(Video).filter(Video.douyin_url == test_url).delete()
        self.db.commit()

        v = Video(video_id="V_DUP_SERP", douyin_url=test_url, status="FOUND")
        self.db.add(v)
        self.db.commit()

        try:
            mock_data = {
                "organic_results": [
                    {"link": test_url, "title": "Already in DB"},
                    {"link": "https://www.douyin.com/video/7461911073162791888", "title": "New Video"}
                ]
            }
            with patch("httpx.Client.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = mock_data
                mock_get.return_value = mock_resp

                res = self.serpapi_provider.search("test", db=self.db)
                self.assertEqual(len(res["videos"]), 2)
                self.assertTrue(res["videos"][0]["already_imported"])
                self.assertFalse(res["videos"][1]["already_imported"])
        finally:
            self.db.query(Video).filter(Video.douyin_url == test_url).delete()
            self.db.commit()

    # 10. Result limit
    def test_10_result_limit(self):
        mock_data = {
            "organic_results": [
                {"link": f"https://www.douyin.com/video/74619110731627910{i:02d}", "title": f"Vid {i}"}
                for i in range(15)
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("limit test", limit=5)
            self.assertEqual(len(res["videos"]), 5)

    # 11. Fewer results than requested -> no second request
    def test_11_fewer_results_than_requested_no_second_request(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.douyin.com/video/7461911073162791001", "title": "Vid 1"},
                {"link": "https://www.douyin.com/video/7461911073162791002", "title": "Vid 2"},
                {"link": "https://www.douyin.com/video/7461911073162791003", "title": "Vid 3"}
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            # Request limit=10, but API only returns 3 valid items
            res = self.serpapi_provider.search("test", limit=10)
            self.assertEqual(len(res["videos"]), 3)
            # Must strictly have been called only ONCE
            self.assertEqual(mock_get.call_count, 1)

    # 12. Exactly one HTTP request per user search
    def test_12_exactly_one_http_request_per_user_search(self):
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"organic_results": []}
            mock_get.return_value = mock_resp

            self.serpapi_provider.search("single request test")
            self.assertEqual(mock_get.call_count, 1)

    # 13. Missing API key
    def test_13_missing_api_key(self):
        unconfigured_provider = SerpApiSearchProvider(api_key="")
        with patch("app.config.get_serpapi_api_key", return_value=""):
            with patch.dict("os.environ", {"SERPAPI_API_KEY": ""}, clear=True):
                res = unconfigured_provider.search("test")
                self.assertFalse(res["success"])
                self.assertEqual(res["state"], "NOT_CONFIGURED")
                self.assertIn("Chưa cấu hình SerpApi", res["message"])

    # 14. Invalid key
    def test_14_invalid_key(self):
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = "Invalid API key"
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("test")
            self.assertFalse(res["success"])
            self.assertEqual(res["state"], "INVALID_KEY")
            self.assertIn("không hợp lệ", res["message"])

    # 15. Quota exceeded
    def test_15_quota_exceeded(self):
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.text = "You have run out of searches"
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("test")
            self.assertFalse(res["success"])
            self.assertEqual(res["state"], "QUOTA_EXCEEDED")
            self.assertIn("quota", res["message"].lower())

    # 16. Network error
    def test_16_network_error(self):
        import httpx
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection failed")):
            res = self.serpapi_provider.search("test")
            self.assertFalse(res["success"])
            self.assertEqual(res["state"], "NETWORK_ERROR")
            self.assertIn("Lỗi kết nối", res["message"])

    # 17. Empty organic results
    def test_17_empty_organic_results(self):
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"organic_results": []}
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("rare keyword")
            self.assertTrue(res["success"])
            self.assertEqual(res["state"], "NO_RESULTS")
            self.assertEqual(len(res["videos"]), 0)
            self.assertIn("Không tìm thấy video Douyin", res["message"])

    # 18. Optional metadata missing
    def test_18_optional_metadata_missing(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.douyin.com/video/7461911073162791001"}  # Missing title, thumbnail, rich_snippet
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            res = self.serpapi_provider.search("test")
            self.assertEqual(len(res["videos"]), 1)
            v = res["videos"][0]
            self.assertEqual(v["video_id"], "7461911073162791001")
            self.assertEqual(v["title"], "Douyin Video 7461911073162791001")
            self.assertEqual(v["thumbnail_url"], "")
            self.assertEqual(v["source"], "serpapi")

    # 19. Cache hit -> zero HTTP requests
    def test_19_cache_hit_zero_http_requests(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.douyin.com/video/7461911073162791001", "title": "Cache Test"}
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            # First search -> issues HTTP request
            res1 = self.serpapi_provider.search("cached item")
            self.assertEqual(mock_get.call_count, 1)

            # Second identical search -> must hit cache with 0 additional HTTP calls!
            res2 = self.serpapi_provider.search("cached item")
            self.assertEqual(mock_get.call_count, 1)
            self.assertTrue(res2.get("cached"))
            self.assertEqual(len(res2["videos"]), 1)

    # 20. Explicit fresh search -> one request
    def test_20_explicit_fresh_search_one_request(self):
        mock_data = {
            "organic_results": [
                {"link": "https://www.douyin.com/video/7461911073162791001", "title": "Fresh Test"}
            ]
        }
        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_data
            mock_get.return_value = mock_resp

            res1 = self.serpapi_provider.search("refresh item")
            self.assertEqual(mock_get.call_count, 1)

            # Force refresh bypasses cache -> calls HTTP again
            res2 = self.serpapi_provider.search("refresh item", force_refresh=True)
            self.assertEqual(mock_get.call_count, 2)

    # 21. Browser provider remains functional
    def test_21_browser_provider_remains_functional(self):
        bp = BrowserSearchProvider()
        self.assertEqual(bp.provider_id, "browser")
        self.assertTrue(bp.requires_douyin_login)

        with patch("app.services.douyin_browser_service.douyin_browser_service.search", return_value={"success": True, "videos": []}):
            res = bp.search("browser kw")
            self.assertTrue(res["success"])
            self.assertEqual(res["provider"], "browser")

    # 22. SnapTikTok workflow unchanged
    def test_22_snaptiktok_workflow_unchanged(self):
        downloader = DownloaderService()
        self.assertTrue(hasattr(downloader, "download_video"))
        self.assertTrue(hasattr(downloader, "download_and_attach"))

    # 23. No provider auto-fallback
    def test_23_no_provider_auto_fallback(self):
        # When SerpApi fails, it should NOT silently invoke browser or other providers
        unconfigured_provider = SerpApiSearchProvider(api_key="")
        with patch("app.services.douyin_browser_service.douyin_browser_service.search") as mock_browser_search:
            res = unconfigured_provider.search("test")
            self.assertFalse(res["success"])
            self.assertEqual(res["state"], "NOT_CONFIGURED")
            mock_browser_search.assert_not_called()

    # 24. Secret redaction
    def test_24_secret_redaction(self):
        secret_key = "super_secret_serpapi_key_99999"
        provider = SerpApiSearchProvider(api_key=secret_key)
        # If server returns error echoing the key
        raw_error = f"Error processing query with key {secret_key}"
        sanitized = provider._sanitize_error(raw_error, secret_key)
        self.assertNotIn(secret_key, sanitized)
        self.assertIn("[REDACTED]", sanitized)


if __name__ == "__main__":
    unittest.main()
