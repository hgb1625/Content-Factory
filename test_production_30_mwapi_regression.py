"""
Regression test suite reproducing and verifying the Production 30 -> MWAPI SQLite configuration integration.

Guarantees:
1. .env MWAPI key is absent.
2. SQLite contains active_ai_provider=mwapi, mwapi_api_key=fake-sqlite-key, mwapi_model=claude-sonnet-4-6.
3. Settings/Test Connection can resolve the key.
4. Production 30 Research (via orchestrator and batch manager) uses the exact SQLite key and model.
5. Outbound HTTP request receives Bearer <fake-sqlite-key> and model <claude-sonnet-4-6>.
6. No "MWAPI API key is not configured" error is raised.
7. Changing the SQLite key dynamically updates subsequent batches without restarting.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Setting, Product
from app.services.ai.manager import get_ai_manager
from app.services.pipeline_orchestrator import (
    ProductionPipelineOrchestrator,
    PipelineBatchManager,
    PipelineState
)


class TestProduction30MWAPIRegression(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary test directory & SQLite database
        self.test_dir = tempfile.mkdtemp(prefix="test_prod30_mwapi_")
        self.db_path = Path(self.test_dir) / "test_prod.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

        # Patch environment to ensure NO MWAPI credentials in .env or os.environ
        self.env_patcher = patch.dict(os.environ, {
            "MWAPI_API_KEY": "",
            "MWAPI_MODEL": "",
            "ACTIVE_AI_PROVIDER": ""
        }, clear=False)
        self.env_patcher.start()

        # Patch SessionLocal across modules to use our test DB
        self.patchers = [
            patch("app.database.SessionLocal", self.SessionLocal),
            patch("app.services.pipeline_orchestrator.SessionLocal", self.SessionLocal),
        ]
        for p in self.patchers:
            p.start()

        # Seed SQLite settings with fake credentials
        self._set_setting("active_ai_provider", "mwapi")
        self._set_setting("mwapi_api_key", "sk-fake-sqlite-key-dd74-test")
        self._set_setting("mwapi_model", "claude-sonnet-4-6")

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.env_patcher.stop()
        self.db.close()
        self.engine.dispose()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _set_setting(self, key: str, value: str):
        rec = self.db.query(Setting).filter(Setting.key == key).first()
        if not rec:
            rec = Setting(key=key, value=value)
            self.db.add(rec)
        else:
            rec.value = value
        self.db.commit()

    def test_01_settings_test_connection_resolves_sqlite_key(self):
        """Settings / Test Connection resolves fake-sqlite-key from SQLite when .env is empty."""
        ai_mgr = get_ai_manager()
        captured_requests = []

        def mock_client_get(url, headers=None, **kwargs):
            captured_requests.append({"url": url, "headers": headers})
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "claude-sonnet-4-6"}]}
            return mock_resp

        with patch("httpx.Client.get", side_effect=mock_client_get):
            res = ai_mgr.test_connection(provider_id="mwapi", db=self.db)
            self.assertTrue(res.get("connected"), f"Test connection should succeed: {res}")
            self.assertTrue(res.get("configured"))
            self.assertEqual(res.get("configured_model"), "claude-sonnet-4-6")

        self.assertGreaterEqual(len(captured_requests), 1)
        auth_hdr = captured_requests[0]["headers"].get("Authorization", "")
        self.assertEqual(auth_hdr, "Bearer sk-fake-sqlite-key-dd74-test")

    def test_02_production_30_research_uses_sqlite_key_and_model(self):
        """Production 30 Research path resolves exact SQLite key and model without .env or fallback."""
        captured_post_requests = []

        fake_mwapi_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "nv": "Hộp bút đa năng thông minh",
                                "nc": "多功能文具盒",
                                "dk": "学生文具盒 创意收纳",
                                "ca": "Hộp bút sức chứa lớn cho học sinh",
                                "h": "Chiếc hộp bút hot nhất mùa tựu trường!"
                            }
                        ])
                    }
                }
            ]
        }

        def mock_client_post(url, json=None, headers=None, **kwargs):
            captured_post_requests.append({
                "url": url,
                "json": json,
                "headers": headers
            })
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = json_lib.dumps(fake_mwapi_response) if "json_lib" in globals() else ""
            mock_resp.json.return_value = fake_mwapi_response
            return mock_resp

        import json as json_lib

        with patch("httpx.Client.post", side_effect=mock_client_post):
            orchestrator = ProductionPipelineOrchestrator(db=self.db)
            products, err = orchestrator._execute_research_stage(
                session=self.db,
                niche="đồ dùng học tập",
                target_count=1,
                force_fresh=True
            )

        self.assertIsNone(err, f"Research stage should succeed without errors: {err}")
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].name_vietnamese, "Hộp bút đa năng thông minh")

        # Verify outbound network call
        self.assertEqual(len(captured_post_requests), 1, "Exactly one MWAPI request should be made")
        outbound = captured_post_requests[0]
        self.assertIn("https://api.mwapi.dev", outbound["url"])

        # Assert outbound request received fake-sqlite-key
        auth_header = outbound["headers"].get("Authorization", "")
        self.assertEqual(auth_header, "Bearer sk-fake-sqlite-key-dd74-test")

        # Assert selected model is claude-sonnet-4-6
        self.assertEqual(outbound["json"].get("model"), "claude-sonnet-4-6")

    def test_03_sqlite_key_update_without_restart(self):
        """Updating SQLite key immediately changes the token used by subsequent Production 30 research."""
        captured_keys = []

        fake_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "nv": "Thước kẻ gập thông minh",
                                "nc": "折叠尺",
                                "dk": "学生折叠尺",
                                "ca": "Thước kẻ tiện dụng cho học sinh",
                                "h": "Cây thước kẻ không thể thiếu!"
                            }
                        ])
                    }
                }
            ]
        }

        def mock_post(url, json=None, headers=None, **kwargs):
            captured_keys.append(headers.get("Authorization", ""))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = fake_response
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            orch = ProductionPipelineOrchestrator(db=self.db)

            # Batch 1 with initial key
            orch._execute_research_stage(session=self.db, niche="học tập", target_count=1, force_fresh=True)
            self.assertEqual(captured_keys[-1], "Bearer sk-fake-sqlite-key-dd74-test")

            # Update key in SQLite on the fly
            self._set_setting("mwapi_api_key", "sk-new-runtime-key-updated-8888")

            # Batch 2 must immediately use the new key
            orch._execute_research_stage(session=self.db, niche="học sinh", target_count=1, force_fresh=True)
            self.assertEqual(captured_keys[-1], "Bearer sk-new-runtime-key-updated-8888")

    def test_04_batch_manager_background_thread_uses_fresh_session_and_sqlite_key(self):
        """PipelineBatchManager worker thread opens its own fresh session and resolves SQLite key/model."""
        captured_requests = []
        fake_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "nv": "Bút chì kim tự bấm",
                                "nc": "自动铅笔",
                                "dk": "学生自动铅笔",
                                "ca": "Bút chì tiện lợi",
                                "h": "Viết êm trơn tru cả ngày!"
                            }
                        ])
                    }
                }
            ]
        }

        def mock_post(url, json=None, headers=None, **kwargs):
            captured_requests.append({"url": url, "json": json, "headers": headers})
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = fake_response
            return mock_resp

        mgr = PipelineBatchManager()
        mgr.batches["BATCH_TEST_BG_01"] = {
            "batch_id": "BATCH_TEST_BG_01",
            "state": "CREATED",
            "status": "RUNNING",
            "counts": {},
            "failures": []
        }

        with patch("httpx.Client.post", side_effect=mock_post), \
             patch.object(ProductionPipelineOrchestrator, "_execute_source_acquisition_stage", return_value=([], [])), \
             patch.object(ProductionPipelineOrchestrator, "_execute_download_stage", return_value=([], [])), \
             patch.object(ProductionPipelineOrchestrator, "_execute_subtitles_and_audio_stage", return_value=([], [])), \
             patch.object(ProductionPipelineOrchestrator, "_execute_rendering_stage", return_value=([], [])), \
             patch.object(ProductionPipelineOrchestrator, "_execute_content_generation_stage", return_value=([], [])):

            # Directly invoke _run_worker synchronously for test determinism
            mgr._run_worker(
                batch_id="BATCH_TEST_BG_01",
                niche="bút chì",
                norm_niche="but chi",
                target_count=1,
                source_provider_id="mock_source",
                force_fresh_research=True,
                mock_content_response=None
            )

        self.assertGreaterEqual(len(captured_requests), 1)
        outbound = captured_requests[0]
        self.assertEqual(outbound["headers"].get("Authorization"), "Bearer sk-fake-sqlite-key-dd74-test")
        self.assertEqual(outbound["json"].get("model"), "claude-sonnet-4-6")

        status = mgr.get_status("BATCH_TEST_BG_01")
        self.assertIsNotNone(status)
        # Should not have failed at RESEARCHING
        research_failures = [f for f in status.get("failures", []) if f.get("stage") == PipelineState.RESEARCHING]
        self.assertEqual(len(research_failures), 0, f"No research stage failures expected: {research_failures}")


if __name__ == "__main__":
    unittest.main()
