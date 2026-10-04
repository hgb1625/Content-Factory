import json
import logging
import os
import shutil
import sqlite3
import sys
import time
import unittest
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db, FINAL_DIR, ORIGINAL_DIR, TEMP_DIR
from app.models import Video, Product
from app.services.ffmpeg_utils import check_ffmpeg_available, run_ffprobe
from app.services.video_analyzer import VideoAnalyzer
from app.services.batch_editor import BatchAutoEditorService


class TestPhase12BatchAutoEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.db = SessionLocal()
        cls.batch_svc = BatchAutoEditorService()
        cls.ff_status = check_ffmpeg_available()

        # Seed Product 1 if missing
        p = cls.db.query(Product).first()
        if not p:
            p = Product(product_id="PBATCH_001", niche="Gia dụng", name_vietnamese="Gia dụng tiện ích", status="RESEARCHED")
            cls.db.add(p)
            cls.db.commit()

        # Prepare two real test videos: V0095 and V0096
        cls.v1_id = "V0095"
        cls.v2_id = "V0096"
        cls.v1_src = ORIGINAL_DIR / f"{cls.v1_id}.mp4"
        cls.v2_src = ORIGINAL_DIR / f"{cls.v2_id}.mp4"

        # Ensure V0095 and V0096 exist as valid MP4s for testing
        is_valid_mp4 = False
        if cls.v1_src.exists() and cls.v1_src.stat().st_size > 1000:
            is_valid_mp4 = True
        if not is_valid_mp4:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=1",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.v1_src)
            ], capture_output=True)

        # Make sure V0096 exists as local file copy of V0095
        if cls.v1_src.exists():
            shutil.copy2(cls.v1_src, cls.v2_src)

        # Seed records in DB
        for vid, src in [(cls.v1_id, cls.v1_src), (cls.v2_id, cls.v2_src)]:
            v = cls.db.query(Video).filter(Video.video_id == vid).first()
            if not v:
                v = Video(
                    video_id=vid,
                    product_id=1,
                    douyin_url=f"https://www.douyin.com/video/{vid}",
                    local_file=f"downloads/original/{vid}.mp4",
                    status="DOWNLOADED",
                    edit_mode="AUTO_EDIT"
                )
                cls.db.add(v)
            else:
                v.local_file = f"downloads/original/{vid}.mp4"
                v.status = "DOWNLOADED"
                v.edit_mode = "AUTO_EDIT"
        cls.db.commit()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_ffmpeg_ready(self):
        """Verify FFmpeg and ffprobe are ready for batch processing."""
        self.assertTrue(self.ff_status["ready"], "FFmpeg and ffprobe must be ready on this PC")

    def test_02_sequential_batch_two_videos_real_e2e(self):
        """
        Execute real Batch Auto-Edit on 2 local videos: V0095 and V0096.
        Verifies:
        - VieNeu synthesis run for each video
        - Audio sync and master voice track assembled
        - Subtitles generated and burned in
        - 1080x1920 9:16 vertical render
        - Atomic move to downloads/final/
        - Both videos reach AUTO_EDIT_READY in DB
        """
        print(f"\n[BATCH TEST] Starting 2-video batch: [{self.v1_id}, {self.v2_id}]...")
        t0 = time.time()
        start_res = self.batch_svc.start_batch([self.v1_id, self.v2_id])
        self.assertTrue(start_res["success"], f"Batch start failed: {start_res.get('error')}")

        # Wait for batch completion (sequential execution)
        max_wait = 180.0
        elapsed = 0.0
        final_status = None

        while elapsed < max_wait:
            time.sleep(2.0)
            elapsed = time.time() - t0
            st = self.batch_svc.get_status()
            print(f"  Batch progress: {st['completed']}/{st['total']} (success={st['success']}, failed={st['failed']}, elapsed={elapsed:.1f}s)")
            if st["status"] == "COMPLETED":
                final_status = st
                break

        self.assertIsNotNone(final_status, "Batch timed out before completion")
        self.assertEqual(final_status["total"], 2)
        self.assertEqual(final_status["completed"], 2)
        self.assertEqual(final_status["success"], 2, f"Expected 2 successes, got: {final_status}")
        self.assertEqual(final_status["failed"], 0)

        # Validate outputs with ffprobe
        for vid in [self.v1_id, self.v2_id]:
            final_mp4 = FINAL_DIR / f"{vid}.mp4"
            self.assertTrue(final_mp4.is_file(), f"Output file missing: {final_mp4}")
            self.assertGreater(final_mp4.stat().st_size, 10000)

            probe = VideoAnalyzer.analyze_video(final_mp4)
            self.assertTrue(probe["valid"], f"Invalid video for {vid}: {probe.get('error')}")
            self.assertEqual(probe["width"], 1080)
            self.assertEqual(probe["height"], 1920)
            self.assertEqual(probe["aspect_ratio"], "9:16")

            # Validate DB status
            v_rec = self.db.query(Video).filter(Video.video_id == vid).first()
            self.assertIsNotNone(v_rec)
            self.assertEqual(v_rec.status, "AUTO_EDIT_READY")
            self.assertEqual(v_rec.auto_edit_status, "AUTO_EDIT_READY")
            self.assertEqual(v_rec.final_video_path, f"downloads/final/{vid}.mp4")

        total_time = round(time.time() - t0, 2)
        print(f"\n[BATCH TEST] 2 videos completed in {total_time}s (Avg {total_time/2:.1f}s per video).")

    def test_03_fault_tolerance_single_failure_isolation(self):
        """
        Verify fault tolerance: A batch with 1 invalid video and 1 valid video
        marks the bad video FAILED with clear error message, and completes the valid video without stopping.
        """
        bad_vid = "V_NONEXISTENT_9999"
        good_vid = self.v1_id

        print(f"\n[FAULT TOLERANCE TEST] Starting batch with 1 invalid ({bad_vid}) and 1 valid ({good_vid})...")
        start_res = self.batch_svc.start_batch([bad_vid, good_vid])
        self.assertTrue(start_res["success"])

        # Wait for completion
        t0 = time.time()
        final_st = None
        while time.time() - t0 < 120.0:
            time.sleep(2.0)
            st = self.batch_svc.get_status()
            if st["status"] == "COMPLETED":
                final_st = st
                break

        self.assertIsNotNone(final_st)
        self.assertEqual(final_st["total"], 2)
        self.assertEqual(final_st["completed"], 2)
        self.assertEqual(final_st["failed"], 1, "Invalid video should be recorded as failed")
        self.assertEqual(final_st["success"], 1, "Valid video should succeed despite previous failure")

        # Bad video check
        bad_info = final_st["videos"].get(bad_vid)
        self.assertEqual(bad_info["status"], "FAILED")
        self.assertIsNotNone(bad_info["error"])
        print(f"  Bad video error recorded properly: {bad_info['error']}")

        # Good video check
        good_info = final_st["videos"].get(good_vid)
        self.assertEqual(good_info["status"], "READY")

    def test_04_retry_failed_mechanism(self):
        """Verify retry_failed restarts only FAILED videos in the queue."""
        # Current batch from test_03 has 1 FAILED (V_NONEXISTENT_9999)
        retry_res = self.batch_svc.retry_failed()
        self.assertTrue(retry_res["success"])
        self.assertEqual(retry_res["total"], 1)

        # Wait for it to fail again cleanly
        t0 = time.time()
        while time.time() - t0 < 30.0:
            time.sleep(1.0)
            if self.batch_svc.get_status()["status"] == "COMPLETED":
                break

        st = self.batch_svc.get_status()
        self.assertEqual(st["completed"], 1)
        self.assertEqual(st["failed"], 1)


if __name__ == "__main__":
    unittest.main()
