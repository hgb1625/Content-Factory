"""
Automated E2E and Regression Test Suite for 30-Product Production Pipeline

Verifies:
1. Golden Path:
   Any niche -> exactly 30 unique products -> 30 unique sources ->
   30 unique downloaded media files (unique SHA-256) -> 30 subtitle artifacts ->
   30 audio/TTS artifacts -> 30 final 9:16 rendered videos ->
   210 platform-content records -> FINAL_READY.
2. Failure & Edge Cases:
   - AI returns 29 products (< 30) -> fails research, 0 saved, NOT FINAL_READY.
   - AI returns 31 products (> 30) -> trims to 30, succeeds.
   - AI returns duplicate product names (< 30 unique) -> fails research, 0 saved.
   - One video source is duplicated -> rejected, batch INCOMPLETE.
   - Two different URLs point to same canonical source ID -> rejected.
   - Two different URLs produce same media hash (SHA-256) -> rejected, duplicate unlinked.
   - One download fails -> batch INCOMPLETE.
   - One subtitle generation fails -> batch INCOMPLETE.
   - One TTS operation fails -> batch INCOMPLETE.
   - One final render fails -> batch INCOMPLETE.
   - One platform-content generation fails -> batch INCOMPLETE.
   - Pipeline resumed after partial completion.
"""
import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Dict, Any
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Product, Video, Voice, Content, Publishing
from app.services.pipeline_orchestrator import (
    ProductionPipelineOrchestrator,
    PipelineState,
    TARGET_PRODUCT_COUNT,
    PLATFORMS_PER_VIDEO,
    TARGET_CONTENT_RECORDS
)
from app.services.video_source.base import VideoSourceProvider, SourceCandidate, SourceStatus
from app.services.video_source.manager import VideoSourceManager
from app.services.downloader_service import calculate_file_sha256
from app.services.tts.vieneu_provider import create_pcm_wav_file


class MockSourceProvider(VideoSourceProvider):
    """Deterministic mock video source provider for testing."""

    def __init__(self, pid: str = "mock_source", dname: str = "Mock Source"):
        self._provider_id = pid
        self._display_name = dname
        self.duplicate_mode = None  # "same_id", "same_url", "fail_one"
        self.call_count = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def display_name(self) -> str:
        return self._display_name

    def is_configured(self) -> bool:
        return True

    def resolve_source(self, url_or_id: str) -> SourceCandidate:
        return SourceCandidate(
            status=SourceStatus.SOURCE_FOUND,
            provider=self.provider_id,
            canonical_source_id=f"CANON_{url_or_id}",
            canonical_url=f"https://mock.video/{url_or_id}",
            download_url=f"https://mock.video/dl_{url_or_id}.mp4"
        )

    def search_source(
        self,
        query: str,
        niche: str = "",
        product_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> SourceCandidate:
        self.call_count += 1
        idx = self.call_count

        if self.duplicate_mode == "fail_one" and idx == 5:
            return SourceCandidate(
                status=SourceStatus.SOURCE_UNAVAILABLE,
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                error_message="Simulated source acquisition failure"
            )

        if self.duplicate_mode == "same_id" and idx == 5:
            # Duplicate canonical source ID with item 1
            return SourceCandidate(
                status=SourceStatus.SOURCE_FOUND,
                provider=self.provider_id,
                canonical_source_id="CANON_SOURCE_001",
                canonical_url=f"https://mock.video/unique_url_{idx}",
                download_url=f"https://mock.video/dl_{idx:03d}.mp4",
                product_id=product_id
            )

        if self.duplicate_mode == "same_url" and idx == 5:
            # Duplicate canonical URL with item 1
            return SourceCandidate(
                status=SourceStatus.SOURCE_FOUND,
                provider=self.provider_id,
                canonical_source_id=f"CANON_SOURCE_{idx:03d}",
                canonical_url="https://mock.video/url_001",
                download_url="https://mock.video/dl_001.mp4",
                product_id=product_id
            )

        return SourceCandidate(
            status=SourceStatus.SOURCE_FOUND,
            provider=self.provider_id,
            canonical_source_id=f"CANON_SOURCE_{idx:03d}",
            canonical_url=f"https://mock.video/url_{idx:03d}",
            download_url=f"https://mock.video/dl_{idx:03d}.mp4",
            product_id=product_id
        )


class TestProductionPipeline30(unittest.TestCase):
    """Full 30-product production pipeline test suite."""

    def setUp(self):
        # Create isolated temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_prod_pipeline_")
        self.db_path = Path(self.test_dir) / "test_pipeline.db"

        # Create isolated SQLite test DB
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        # Mock mock media files
        self.media_dir = Path(self.test_dir) / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.final_dir = Path(self.test_dir) / "downloads" / "final"
        self.final_dir.mkdir(parents=True, exist_ok=True)

        self.original_dir = Path(self.test_dir) / "downloads" / "original"
        self.original_dir.mkdir(parents=True, exist_ok=True)

        # Patch directories in pipeline orchestrator and database
        self.patchers = [
            patch("app.services.pipeline_orchestrator.FINAL_DIR", self.final_dir),
            patch("app.services.pipeline_orchestrator.ORIGINAL_DIR", self.original_dir),
            patch("app.services.auto_editor.FINAL_DIR", self.final_dir),
            patch("app.services.downloader_service.ORIGINAL_DIR", self.original_dir),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self.session.close()
        self.engine.dispose()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _fake_render(self, db_or_id=None, id_or_db=None, video_id=None, db=None, **kwargs):
        vid = video_id or (db_or_id if isinstance(db_or_id, str) else id_or_db)
        final_f = self.final_dir / f"{vid}.mp4"
        final_f.parent.mkdir(parents=True, exist_ok=True)
        final_f.write_bytes(b"0" * 2000)
        return {"success": True, "video_id": vid, "final_path": str(final_f)}

    def _fake_content(self, db, video_id, **kwargs):
        c = db.query(Content).filter(Content.video_id == video_id).first()
        if not c:
            c = Content(video_id=video_id)
            db.add(c)
        c.facebook_personal_caption = "FB Personal"
        c.facebook_page_caption = "FB Page"
        c.tiktok_caption = "TikTok"
        c.threads_caption = "Threads"
        c.instagram_caption = "Instagram"
        c.shopee_caption = "Shopee"
        c.youtube_title = "YouTube"
        c.status = "CONTENT_READY"
        db.commit()
        return {"success": True, "video_id": video_id, "status": "CONTENT_READY"}

    def _create_mock_media_file(self, filename: str, content: bytes = b"dummy_mp4_header_test") -> Path:
        """Helper to create a mock media file with distinct content for hashing."""
        fpath = self.media_dir / filename
        fpath.write_bytes(content)
        return fpath

    def _generate_mock_products_list(self, count: int, duplicate_name: bool = False) -> list:
        """Generate test products conforming to contract."""
        products = []
        for i in range(1, count + 1):
            name = f"Sản phẩm gia dụng thông minh {i:03d}"
            if duplicate_name and i == 2:
                name = "Sản phẩm gia dụng thông minh 001"  # Duplicate of item 1
            products.append({
                "name_vietnamese": name,
                "name_chinese": f"智能家居产品 {i:03d}",
                "douyin_keywords": f"smart home {i}",
                "content_angle": f"Góc nhìn tiện ích giải pháp cho gia đình {i}",
                "hook": f"Bí quyết dọn nhà cực nhanh số {i} ai cũng cần biết"
            })
        return products

    # -------------------------------------------------------------------------
    # 1. Full E2E Success Test
    # -------------------------------------------------------------------------

    @patch("app.services.content_service.ContentService.generate_content_for_video")
    @patch("app.services.auto_editor.AutoEditorService.render_final_video")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_e2e_30_product_pipeline_success(
        self,
        mock_get_active_provider,
        mock_dl_file,
        mock_validate_media,
        mock_render_video,
        mock_generate_content
    ):
        """
        Golden Path:
        Niche -> Exactly 30 products -> 30 sources -> 30 downloaded ->
        30 subtitles -> 30 audio -> 30 final 9:16 videos ->
        210 platform content records -> FINAL_READY.
        """
        # 1. Mock AI Provider returning exactly 30 valid products
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        # 2. Mock Source Manager with deterministic provider
        source_prov = MockSourceProvider("test_mock_provider")
        src_mgr = VideoSourceManager()
        src_mgr.register_provider(source_prov)

        # 3. Mock Download
        def fake_download(url, dest_path, **kwargs):
            dest_path = Path(dest_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(f"unique_media_content_{url}".encode("utf-8"))
            return True, None
        mock_dl_file.side_effect = fake_download
        mock_validate_media.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)

        # 4. Mock Final Render
        mock_render_video.side_effect = self._fake_render

        # 5. Mock Content Generation (populates Content table with all 7 platforms)
        mock_generate_content.side_effect = self._fake_content

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(
            niche="Đồ gia dụng thông minh",
            target_count=30,
            source_provider_id="test_mock_provider",
            db=self.session
        )

        # Assert full contract
        self.assertEqual(result["status"], PipelineState.FINAL_READY)
        self.assertTrue(result["is_final_ready"])
        self.assertEqual(len(result["failures"]), 0)

        counts = result["counts"]
        self.assertEqual(counts["products"], 30)
        self.assertEqual(counts["unique_products"], 30)
        self.assertEqual(counts["valid_video_sources"], 30)
        self.assertEqual(counts["unique_video_identities"], 30)
        self.assertEqual(counts["unique_media_hashes"], 30)
        self.assertEqual(counts["downloaded_valid_videos"], 30)
        self.assertEqual(counts["subtitle_artifacts"], 30)
        self.assertEqual(counts["audio_artifacts"], 30)
        self.assertEqual(counts["final_videos"], 30)
        self.assertEqual(counts["platform_content_records"], 210)

    # -------------------------------------------------------------------------
    # 2. Failure Cases
    # -------------------------------------------------------------------------

    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_ai_returns_29_products(self, mock_get_active_provider):
        """AI returns 29 products (< 30) -> Fails research, 0 saved, NOT FINAL_READY."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(29)
        mock_get_active_provider.return_value = mock_ai

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, db=self.session)

        self.assertNotEqual(result["status"], PipelineState.FINAL_READY)
        self.assertEqual(result["status"], PipelineState.FAILED)
        self.assertFalse(result["is_final_ready"])
        # Persisted products must be 0
        db_prods = self.session.query(Product).count()
        self.assertEqual(db_prods, 0)

    @patch("app.services.content_service.ContentService.generate_content_for_video")
    @patch("app.services.auto_editor.AutoEditorService.render_final_video")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_ai_returns_31_products_trims_to_30(
        self, mock_get_active_provider, mock_dl, mock_val, mock_rend, mock_cont
    ):
        """AI returns 31 products (> 30) -> Deterministically trims to 30 and succeeds."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(31)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_trim_provider")
        VideoSourceManager().register_provider(source_prov)

        mock_dl.side_effect = lambda u, p, **kw: (Path(p).write_bytes(f"media_{u}".encode()) or True, None)
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)
        mock_rend.side_effect = self._fake_render
        mock_cont.side_effect = self._fake_content

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_trim_provider", db=self.session)

        self.assertEqual(result["status"], PipelineState.FINAL_READY)
        self.assertEqual(result["counts"]["products"], 30)

    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_duplicate_product_names(self, mock_get_active_provider):
        """AI returns duplicate product names (unique < 30) -> Fails research, 0 saved."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30, duplicate_name=True)
        mock_get_active_provider.return_value = mock_ai

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, db=self.session)

        self.assertNotEqual(result["status"], PipelineState.FINAL_READY)
        self.assertEqual(result["status"], PipelineState.FAILED)
        self.assertEqual(self.session.query(Product).count(), 0)

    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_duplicate_video_source(self, mock_get_active_provider):
        """One video source is duplicated -> Rejected, batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_dup_source")
        source_prov.duplicate_mode = "same_id"
        VideoSourceManager().register_provider(source_prov)

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_dup_source", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_different_urls_same_canonical_source_id(self, mock_get_active_provider):
        """Two different URLs point to the same canonical source ID -> Rejected."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_same_canon_id")
        source_prov.duplicate_mode = "same_id"  # Returns same canonical ID under different URL
        VideoSourceManager().register_provider(source_prov)

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_same_canon_id", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_duplicate_media_hash(self, mock_get_active_provider, mock_dl, mock_val):
        """Two different source URLs produce identical media hash -> Rejected, batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_dup_hash_provider")
        VideoSourceManager().register_provider(source_prov)

        # Video 1 and Video 2 produce identical bytes
        def fake_dl(url, path, **kw):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if "001" in url or "002" in url:
                p.write_bytes(b"identical_hash_media_bytes")
            else:
                p.write_bytes(f"unique_media_{url}".encode())
            return True, None
        mock_dl.side_effect = fake_dl
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_dup_hash_provider", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])
        has_dup_failure = any("Duplicate media hash" in f.get("error", "") for f in result["failures"])
        self.assertTrue(has_dup_failure)

    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_download_fails(self, mock_get_active_provider, mock_dl):
        """One video download fails -> Batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_dl_fail_prov")
        VideoSourceManager().register_provider(source_prov)

        # Download fails on item 5
        def fake_dl(url, path, **kw):
            if "005" in url:
                return False, "HTTP 500 Network Timeout"
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"media_{url}".encode())
            return True, None
        mock_dl.side_effect = fake_dl

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_dl_fail_prov", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    @patch("app.services.subtitle_service.SubtitleService.generate_srt")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_subtitle_generation_fails(self, mock_get_active_provider, mock_dl, mock_val, mock_srt):
        """One subtitle generation fails -> Batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_sub_fail_prov")
        VideoSourceManager().register_provider(source_prov)

        mock_dl.side_effect = lambda u, p, **kw: (Path(p).write_bytes(f"media_{u}".encode()) or True, None)
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)

        # Subtitle fails on one item
        call_count = [0]
        def fake_srt(segs, path, **kw):
            call_count[0] += 1
            if call_count[0] == 3:
                return False
            Path(path).write_text("1\n00:00:00,000 --> 00:00:05,000\nSub", encoding="utf-8")
            return True
        mock_srt.side_effect = fake_srt

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_sub_fail_prov", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    @patch("app.services.tts.vieneu_provider.create_pcm_wav_file")
    @patch("app.services.tts.vieneu_provider.VieNeuProvider.synthesize")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_tts_fails(self, mock_get_active_provider, mock_dl, mock_val, mock_synth, mock_pcm):
        """One TTS operation fails -> Batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_tts_fail_prov")
        VideoSourceManager().register_provider(source_prov)

        mock_dl.side_effect = lambda u, p, **kw: (Path(p).write_bytes(f"media_{u}".encode()) or True, None)
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)

        # TTS fails
        mock_synth.return_value = {"success": False, "error": "TTS Engine Crash"}
        mock_pcm.return_value = False  # Both fail

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_tts_fail_prov", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    @patch("app.services.auto_editor.AutoEditorService.render_final_video")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_rendering_fails(self, mock_get_active_provider, mock_dl, mock_val, mock_rend):
        """One final video render fails -> Batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_render_fail_prov")
        VideoSourceManager().register_provider(source_prov)

        mock_dl.side_effect = lambda u, p, **kw: (Path(p).write_bytes(f"media_{u}".encode()) or True, None)
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)

        render_count = [0]
        def fake_r(db_or_id=None, id_or_db=None, video_id=None, **kw):
            render_count[0] += 1
            if render_count[0] == 4:
                return {"success": False, "error": "FFmpeg Filter Graph Error"}
            return self._fake_render(db_or_id=db_or_id, id_or_db=id_or_db, video_id=video_id)
        mock_rend.side_effect = fake_r

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_render_fail_prov", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    @patch("app.services.content_service.ContentService.generate_content_for_video")
    @patch("app.services.auto_editor.AutoEditorService.render_final_video")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_failure_content_generation_fails(self, mock_get_active_provider, mock_dl, mock_val, mock_rend, mock_cont):
        """One platform content generation fails -> Batch INCOMPLETE."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_content_fail_prov")
        VideoSourceManager().register_provider(source_prov)

        mock_dl.side_effect = lambda u, p, **kw: (Path(p).write_bytes(f"media_{u}".encode()) or True, None)
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)
        mock_rend.side_effect = self._fake_render

        content_calls = [0]
        def fake_c(db=None, video_id=None, vid=None, **kw):
            content_calls[0] += 1
            if content_calls[0] == 7:
                return {"success": False, "error": "AI Quota Exceeded on Content"}
            return self._fake_content(db, video_id or vid)
        mock_cont.side_effect = fake_c

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        result = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_content_fail_prov", db=self.session)

        self.assertEqual(result["status"], PipelineState.INCOMPLETE)
        self.assertFalse(result["is_final_ready"])

    # -------------------------------------------------------------------------
    # 3. Resumption Test
    # -------------------------------------------------------------------------

    @patch("app.services.content_service.ContentService.generate_content_for_video")
    @patch("app.services.auto_editor.AutoEditorService.render_final_video")
    @patch("app.services.pipeline_orchestrator.validate_downloaded_media")
    @patch("app.services.downloader_service.SnapTikTokDownloader.download_file")
    @patch("app.services.ai.manager.AIProviderManager.get_active_provider")
    def test_pipeline_resumption_after_partial_completion(
        self, mock_get_active_provider, mock_dl, mock_val, mock_rend, mock_cont
    ):
        """Pipeline resumed after partial completion reuses valid artifacts and reaches FINAL_READY."""
        mock_ai = MagicMock()
        mock_ai.generate_products.return_value = self._generate_mock_products_list(30)
        mock_get_active_provider.return_value = mock_ai

        source_prov = MockSourceProvider("mock_resume_prov")
        VideoSourceManager().register_provider(source_prov)

        # Fail on initial run during download
        should_fail = [True]
        def fake_dl(url, path, **kw):
            if should_fail[0] and "010" in url:
                return False, "Interrupted"
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"media_{url}".encode())
            return True, None
        mock_dl.side_effect = fake_dl
        mock_val.return_value = (True, {"duration": 5.0, "width": 1080, "height": 1920}, None)
        mock_rend.side_effect = self._fake_render
        mock_cont.side_effect = self._fake_content

        orchestrator = ProductionPipelineOrchestrator(db=self.session)
        res1 = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_resume_prov", db=self.session)
        self.assertEqual(res1["status"], PipelineState.INCOMPLETE)

        # Second run: Resume with fix
        should_fail[0] = False
        res2 = orchestrator.run_pipeline(niche="Gia dụng", target_count=30, source_provider_id="mock_resume_prov", db=self.session)
        self.assertEqual(res2["status"], PipelineState.FINAL_READY)
        self.assertTrue(res2["is_final_ready"])


if __name__ == "__main__":
    unittest.main()
