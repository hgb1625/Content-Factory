"""
Regression Test Suite for Pexels API Key Configuration & Production 30 Stage 2 Integration.

Verifies:
1. SQLite overrides .env (SQLite is authoritative).
2. Save new Pexels key via Settings replaces SQLite value.
3. Blank input preserves existing key.
4. Masked placeholder (••••xxxx or ****xxxx) preserves existing key.
5. Test Connection uses SQLite key and sends lightweight request.
6. Production 30 Stage 2 (Video Source Discovery) uses SQLite key.
7. Key change in SQLite takes effect immediately on next batch without restart or code modification.
8. Secret key never appears in logs or error messages.
"""
import os
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import httpx

from app.database import Base
from app.models import Setting, Product, Video
from app.config import get_pexels_api_key, get_key_hint, mask_api_key
from app.services.video_source.manager import VideoSourceManager
from app.services.video_source.providers.pexels import PexelsSourceProvider
from app.services.video_source.base import SourceStatus
from app.routes.settings import save_settings, get_current_settings


class TestPexelsSettingsRegression(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary file SQLite database for thread safety
        self.temp_dir = tempfile.mkdtemp(prefix="test_pexels_")
        self.db_path = Path(self.temp_dir) / "test.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False}, echo=False)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_sqlite_overrides_env(self):
        """SQLite pexels_api_key must take priority over .env PEXELS_API_KEY."""
        # Setup: .env has key, SQLite has key
        with patch.dict(os.environ, {"PEXELS_API_KEY": "fake-env-key-9999"}):
            self.db.add(Setting(key="pexels_api_key", value="fake-sqlite-key-1111"))
            self.db.commit()

            # 1. With db session passed
            key_resolved = get_pexels_api_key(db=self.db)
            self.assertEqual(key_resolved, "fake-sqlite-key-1111", "SQLite must override .env when db passed")

            # 2. With SessionLocal patch (db=None)
            with patch("app.database.SessionLocal", self.SessionLocal):
                key_resolved_none = get_pexels_api_key(db=None)
                self.assertEqual(key_resolved_none, "fake-sqlite-key-1111", "SQLite must override .env when db=None")

            # 3. If SQLite has no key, fallback to .env occurs
            self.db.query(Setting).filter(Setting.key == "pexels_api_key").delete()
            self.db.commit()

            key_fallback = get_pexels_api_key(db=self.db)
            self.assertEqual(key_fallback, "fake-env-key-9999", "Fallback to .env only when SQLite is empty")

    def test_02_save_new_key_blank_and_masked_preservation(self):
        """Entering new key saves to SQLite; blank or masked preserves existing."""
        from fastapi import Request
        mock_request = MagicMock(spec=Request)

        # 1. Save new key
        with patch("app.routes.settings.templates.TemplateResponse"), \
             patch("app.routes.settings.SubtitleService.save_user_settings"), \
             patch("app.routes.settings.set_key"):
            save_settings(
                request=mock_request,
                pexels_api_key="  fake-sqlite-key-2222  ",
                db=self.db
            )

        row = self.db.query(Setting).filter(Setting.key == "pexels_api_key").first()
        self.assertIsNotNone(row)
        self.assertEqual(row.value, "fake-sqlite-key-2222", "New key should be trimmed and stored")

        # 2. Blank preserves existing key
        with patch("app.routes.settings.templates.TemplateResponse"), \
             patch("app.routes.settings.SubtitleService.save_user_settings"), \
             patch("app.routes.settings.set_key"):
            save_settings(
                request=mock_request,
                pexels_api_key="   ",
                db=self.db
            )

        row = self.db.query(Setting).filter(Setting.key == "pexels_api_key").first()
        self.assertEqual(row.value, "fake-sqlite-key-2222", "Blank input must preserve existing key")

        # 3. Masked placeholder preserves existing key
        with patch("app.routes.settings.templates.TemplateResponse"), \
             patch("app.routes.settings.SubtitleService.save_user_settings"), \
             patch("app.routes.settings.set_key"):
            save_settings(
                request=mock_request,
                pexels_api_key="••••2222",
                db=self.db
            )

        row = self.db.query(Setting).filter(Setting.key == "pexels_api_key").first()
        self.assertEqual(row.value, "fake-sqlite-key-2222", "Masked placeholder (••••) must preserve existing key")

        with patch("app.routes.settings.templates.TemplateResponse"), \
             patch("app.routes.settings.SubtitleService.save_user_settings"), \
             patch("app.routes.settings.set_key"):
            save_settings(
                request=mock_request,
                pexels_api_key="****2222",
                db=self.db
            )

        row = self.db.query(Setting).filter(Setting.key == "pexels_api_key").first()
        self.assertEqual(row.value, "fake-sqlite-key-2222", "Masked placeholder (****) must preserve existing key")

    def test_03_test_connection_uses_sqlite_key(self):
        """Settings Test Connection must make request using SQLite key."""
        self.db.add(Setting(key="pexels_api_key", value="fake-sqlite-test-key"))
        self.db.commit()

        provider = PexelsSourceProvider()
        captured_headers = {}

        def mock_get(url, headers=None, params=None):
            captured_headers.update(headers or {})
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"page": 1, "per_page": 1, "total_results": 100, "videos": []}
            return mock_resp

        with patch("httpx.Client.get", side_effect=mock_get):
            res = provider.test_connection(db=self.db)

        self.assertTrue(res["connected"])
        self.assertTrue(res["success"])
        self.assertEqual(captured_headers.get("Authorization"), "fake-sqlite-test-key")

    def test_04_production_30_stage_2_uses_sqlite_key(self):
        """Production 30 Stage 2 (Video Source Discovery) must use SQLite key."""
        self.db.add(Setting(key="pexels_api_key", value="fake-sqlite-stage2-key"))
        self.db.commit()

        prod = Product(
            product_id="P0001",
            niche="Gia dụng",
            name_vietnamese="Hộp cơm giữ nhiệt",
            douyin_keywords="hop com giu nhiet",
            status="RESEARCHED"
        )
        self.db.add(prod)
        self.db.commit()

        captured_auth = []

        def mock_search_get(url, headers=None, params=None):
            captured_auth.append((headers or {}).get("Authorization"))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "page": 1,
                "per_page": 5,
                "total_results": 1,
                "videos": [{
                    "id": 987654,
                    "width": 1080,
                    "height": 1920,
                    "duration": 15,
                    "url": "https://www.pexels.com/video/987654/",
                    "video_files": [{
                        "id": 111,
                        "quality": "hd",
                        "file_type": "video/mp4",
                        "width": 1080,
                        "height": 1920,
                        "link": "https://test.pexels.com/download/987654.mp4"
                    }]
                }]
            }
            return mock_resp

        with patch("httpx.Client.get", side_effect=mock_search_get):
            manager = VideoSourceManager()
            result = manager.acquire_batch_sources(
                products=[prod],
                provider_id="pexels",
                db=self.db
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["acquired_count"], 1)
        self.assertEqual(captured_auth, ["fake-sqlite-stage2-key"])
        candidate = result["candidates"][0]
        self.assertEqual(candidate.canonical_source_id, "pexels:987654")

    def test_05_key_change_takes_effect_without_source_modification(self):
        """Updating SQLite key must take effect immediately on next call without restarting."""
        setting_row = Setting(key="pexels_api_key", value="initial-key-111")
        self.db.add(setting_row)
        self.db.commit()

        provider = PexelsSourceProvider()
        self.assertEqual(provider.get_api_key(db=self.db), "initial-key-111")

        # User updates key in Settings
        setting_row.value = "updated-key-222"
        self.db.commit()

        self.assertEqual(provider.get_api_key(db=self.db), "updated-key-222", "Provider must read updated key from SQLite immediately")

    def test_06_secret_does_not_appear_in_logs_or_errors(self):
        """Secret key should never leak into error messages or responses."""
        secret_key = "super-secret-pexels-key-xyz"
        self.db.add(Setting(key="pexels_api_key", value=secret_key))
        self.db.commit()

        provider = PexelsSourceProvider()

        # Simulate 401 Unauthorized
        def mock_401(url, headers=None, params=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = f"Unauthorized error for key {secret_key}"
            return mock_resp

        with patch("httpx.Client.get", side_effect=mock_401):
            res = provider.test_connection(db=self.db)

        self.assertFalse(res["connected"])
        # Ensure secret_key is not in any field
        for k, v in res.items():
            self.assertNotIn(secret_key, str(v), f"Secret key leaked in test_connection result field '{k}'")

        # Simulate error in search_source
        with patch("httpx.Client.get", side_effect=mock_401):
            cand = provider.search_source(query="test", db=self.db)

        self.assertEqual(cand.status, SourceStatus.SOURCE_UNAVAILABLE)
        self.assertNotIn(secret_key, cand.error_message, "Secret key leaked in search_source error_message")

    def test_07_testclient_endpoints_and_html_render(self):
        """Test FastAPI /settings and /api/settings/test-pexels endpoints."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.database import get_db

        def override_get_db():
            try:
                yield self.db
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db

        try:
            client = TestClient(app)

            # 1. GET /settings with saved key
            self.db.add(Setting(key="pexels_api_key", value="pexels-client-test-key-5555"))
            self.db.commit()

            resp = client.get("/settings")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("input_key_pexels", resp.text)
            self.assertIn("btnTestPexels", resp.text)
            self.assertIn("Pexels Video API", resp.text)
            self.assertIn("••••5555", resp.text)

            # 2. POST /api/settings/test-pexels (HTTP 200 mock)
            def mock_get(url, headers=None, params=None):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"page": 1, "per_page": 1, "total_results": 42}
                return mock_resp

            with patch("httpx.Client.get", side_effect=mock_get):
                test_resp = client.post("/api/settings/test-pexels", json={})
                self.assertEqual(test_resp.status_code, 200)
                data = test_resp.json()
                self.assertTrue(data["connected"])
                self.assertIn("thành công", data["message"].lower())
                self.assertEqual(data["key_hint"], "••••5555")

        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
