"""
Test Production Pipeline API Routes and Background Manager.

Verifies:
1. GET /pipeline returns 200 HTML page.
2. POST /api/pipeline/run-batch input validation (empty niche, invalid count).
3. POST /api/pipeline/run-batch returns batch_id, status=RUNNING, and defaults strictly to 30.
4. Duplicate concurrent batch for same niche is blocked (returns 409).
5. GET /api/pipeline/status/{batch_id} returns 200 with counts, state, and progress.
6. GET /api/pipeline/active returns batches list.
"""
import unittest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient

from app.main import app
from app.services.pipeline_orchestrator import (
    get_pipeline_batch_manager,
    PipelineBatchManager,
    PipelineState
)


class TestPipelineRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.batch_mgr = get_pipeline_batch_manager()
        with self.batch_mgr.state_lock:
            self.batch_mgr.running_niches.clear()
            self.batch_mgr.batches.clear()

    def tearDown(self):
        with self.batch_mgr.state_lock:
            self.batch_mgr.running_niches.clear()
            self.batch_mgr.batches.clear()

    def test_01_get_pipeline_ui_page(self):
        resp = self.client.get("/pipeline")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("30 Video", resp.text)
        self.assertIn("pipelineForm", resp.text)

    def test_02_run_batch_validation(self):
        # Empty niche -> 400
        resp = self.client.post("/api/pipeline/run-batch", json={"niche": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["success"])

        # Invalid count -> 400
        resp = self.client.post("/api/pipeline/run-batch", json={"niche": "Gia dụng", "product_count": 0})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["success"])

    @patch.object(PipelineBatchManager, "_run_worker")
    def test_03_run_batch_starts_background_with_default_30(self, mock_worker):
        resp = self.client.post("/api/pipeline/run-batch", json={"niche": "Nhà bếp thông minh"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["target_count"], 30)
        self.assertTrue(data["batch_id"].startswith("BATCH_PROD_"))
        self.assertEqual(data["status"], "RUNNING")
        self.assertIn(data["batch_id"], data["poll_url"])

    @patch.object(PipelineBatchManager, "_run_worker")
    def test_04_prevent_duplicate_concurrent_batches_for_same_niche(self, mock_worker):
        # First call succeeds
        resp1 = self.client.post("/api/pipeline/run-batch", json={"niche": "Thú cưng"})
        self.assertEqual(resp1.status_code, 200)
        self.assertTrue(resp1.json()["success"])

        # Immediate second call with identical niche is blocked
        resp2 = self.client.post("/api/pipeline/run-batch", json={"niche": "thú cưng "})
        self.assertEqual(resp2.status_code, 409)
        self.assertFalse(resp2.json()["success"])
        self.assertIn("đang chạy", resp2.json()["error"])

    @patch.object(PipelineBatchManager, "_run_worker")
    def test_05_poll_pipeline_status(self, mock_worker):
        resp = self.client.post("/api/pipeline/run-batch", json={"niche": "Làm đẹp"})
        batch_id = resp.json()["batch_id"]

        status_resp = self.client.get(f"/api/pipeline/status/{batch_id}")
        self.assertEqual(status_resp.status_code, 200)
        b = status_resp.json()["batch"]
        self.assertEqual(b["batch_id"], batch_id)
        self.assertEqual(b["niche"], "Làm đẹp")
        self.assertEqual(b["target_count"], 30)
        self.assertIn("counts", b)
        self.assertIn("products", b["counts"])
        self.assertIn("progress_percentage", b)

        # Non-existent batch -> 404
        not_found = self.client.get("/api/pipeline/status/BATCH_FAKE_999")
        self.assertEqual(not_found.status_code, 404)

    @patch.object(PipelineBatchManager, "_run_worker")
    def test_06_active_batches_endpoint(self, mock_worker):
        self.client.post("/api/pipeline/run-batch", json={"niche": "Thời trang"})
        resp = self.client.get("/api/pipeline/active")
        self.assertEqual(resp.status_code, 200)
        batches = resp.json()["batches"]
        self.assertGreaterEqual(len(batches), 1)
        self.assertEqual(batches[0]["niche"], "Thời trang")


if __name__ == "__main__":
    unittest.main()
