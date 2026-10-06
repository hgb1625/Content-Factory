"""
Offline Unit and Integration Tests for Douyin Browser Prototype (28-Point Verification Suite)

Section 1: Baseline Architecture & Recovery (Tests 1-18)
Section 2: Connect & Login UX Regression Suite (Tests 19-28)
  19. click Connect -> launches browser on port 9333
  20. opens douyin.com
  21. unauthenticated state -> LOGIN_REQUIRED
  22. visible login UI open attempt
  23. missing login button -> graceful LOGIN_REQUIRED
  24. CAPTCHA -> VERIFICATION_REQUIRED
  25. authenticated session -> CONNECTED
  26. unrelated process on 9222 ignored
  27. no duplicate Chrome instances
  28. existing SnapTikTok workflow unchanged
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal, init_db
from app.models import Video, Product
from app.services.douyin_browser_service import (
    DouyinBrowserService,
    DouyinSearchResult,
    detect_douyin_page_state,
    extract_canonical_douyin_video_id,
    parse_douyin_search_items,
    DEFAULT_CDP_PORT,
    STATE_NOT_CONNECTED,
    STATE_BROWSER_OPEN,
    STATE_LOGIN_REQUIRED,
    STATE_VERIFICATION_REQUIRED,
    STATE_CONNECTED,
    STATE_SEARCHING,
    STATE_ERROR,
)
from app.services.douyin_service import DouyinService, extract_douyin_video_id, get_canonical_douyin_url
from app.services.downloader_service import DownloaderService


class TestDouyinBrowserArchitecture(unittest.TestCase):

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.client = TestClient(app)
        self.test_profile = BASE_DIR / "runtime" / "test_douyin_profile"
        self.browser_service = DouyinBrowserService(profile_dir=self.test_profile, cdp_port=9333)

    def tearDown(self):
        self.db.close()

    # 1. Persistent profile reuse
    def test_01_persistent_profile_reuse(self):
        p1 = self.browser_service.profile_dir
        self.assertTrue("douyin_browser_profile" in p1.name or "test_douyin_profile" in p1.name)
        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            res = self.browser_service.open_browser()
            self.assertTrue(res["success"])
            self.assertEqual(self.browser_service.profile_dir, p1)

    # 2. Duplicate Chrome launch prevention
    def test_02_duplicate_chrome_launch_prevention(self):
        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "get_status", return_value={"state": STATE_CONNECTED, "connected": True}):
                res = self.browser_service.open_browser()
                self.assertTrue(res["success"])
                self.assertEqual(res["state"], STATE_CONNECTED)

    # 3. Reconnect after app restart
    def test_03_reconnect_after_app_restart(self):
        fresh_service = DouyinBrowserService(profile_dir=self.test_profile, cdp_port=9333)
        with patch.object(fresh_service, "is_cdp_reachable", return_value=True):
            with patch.object(fresh_service, "get_status", return_value={"state": STATE_CONNECTED, "connected": True}):
                reconn = fresh_service.reconnect_if_alive()
                self.assertTrue(reconn["success"])
                self.assertTrue(reconn["reconnected"])
                self.assertEqual(reconn["state"], STATE_CONNECTED)

    # 4. Dead browser detection
    def test_04_dead_browser_detection(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.pid = 9999
        self.browser_service._process = mock_proc

        is_alive = self.browser_service._check_process_alive()
        self.assertFalse(is_alive)
        self.assertIsNone(self.browser_service._process)

        with patch("httpx.get", side_effect=Exception("Connection refused")):
            self.assertFalse(self.browser_service.is_cdp_reachable())
            status = self.browser_service.get_status()
            self.assertEqual(status["state"], STATE_NOT_CONNECTED)

    # 5. Stale CDP recovery
    def test_05_stale_cdp_recovery(self):
        with patch("websockets.sync.client.connect", side_effect=Exception("WebSocket closed")):
            val = self.browser_service._execute_cdp_eval("ws://127.0.0.1:9333/stale", "1+1")
            self.assertIsNone(val, "Stale CDP socket should be caught safely without unhandled exception")

    # 6. Sequential search lock
    def test_06_sequential_search_lock(self):
        self.browser_service._search_lock.acquire()
        try:
            res = self.browser_service.search("test keyword")
            self.assertFalse(res["success"])
            self.assertEqual(res["state"], STATE_ERROR)
            self.assertIn("tìm kiếm tuần tự", res["error"])
        finally:
            self.browser_service._search_lock.release()

    # 7. Login detection
    def test_07_login_detection(self):
        self.assertEqual(
            detect_douyin_page_state(url="https://passport.douyin.com/login"),
            STATE_LOGIN_REQUIRED
        )
        self.assertEqual(
            detect_douyin_page_state(url="https://www.douyin.com/search/food", dom_text="请先登录，再继续搜索吧"),
            STATE_LOGIN_REQUIRED
        )
        self.assertEqual(
            detect_douyin_page_state(
                url="https://www.douyin.com/search/food",
                dom_html_snippet="<div class='login-guide-container'></div>"
            ),
            STATE_LOGIN_REQUIRED
        )

    # 8. CAPTCHA detection
    def test_08_captcha_detection(self):
        self.assertEqual(
            detect_douyin_page_state(
                url="https://www.douyin.com/search/abc",
                title="验证码中间页",
                dom_text="请完成下列验证后继续"
            ),
            STATE_VERIFICATION_REQUIRED
        )
        self.assertEqual(
            detect_douyin_page_state(
                url="https://www.douyin.com/search/abc",
                dom_text="按住左边按钮拖动完成上方拼图"
            ),
            STATE_VERIFICATION_REQUIRED
        )

    # 9. Bounded result scrolling
    def test_09_bounded_result_scrolling(self):
        with patch.object(self.browser_service, "get_status", return_value={"connected": True, "state": STATE_CONNECTED}):
            with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://mock"}):
                with patch.object(self.browser_service, "_navigate_tab", return_value=True):
                    mock_items = [
                        {"href": "/video/7461911073162791001", "title": "Vid 1"},
                        {"href": "/video/7461911073162791002", "title": "Vid 2"},
                        {"href": "/video/7461911073162791003", "title": "Vid 3"},
                    ]
                    with patch.object(self.browser_service, "_execute_cdp_eval", return_value=mock_items):
                        res = self.browser_service.search("bounded test", limit=3)
                        self.assertTrue(res["success"])
                        self.assertEqual(len(res["videos"]), 3)

    # 10. Canonical video ID extraction
    def test_10_canonical_video_id_extraction(self):
        vid1 = extract_canonical_douyin_video_id("/video/7461911073162791080")
        self.assertEqual(vid1, "7461911073162791080")
        vid2 = extract_canonical_douyin_video_id("https://www.douyin.com/search/test?modal_id=7461911073162791099")
        self.assertEqual(vid2, "7461911073162791099")
        vid_none = extract_canonical_douyin_video_id("https://www.douyin.com/user/12345")
        self.assertIsNone(vid_none)

    # 11. Duplicate result removal
    def test_11_duplicate_result_removal(self):
        mock_raw = [
            {"href": "/video/7461911073162791001", "title": "Card 1"},
            {"href": "/video/7461911073162791001", "title": "Card 1 duplicate"},
            {"href": "/video/7461911073162791002", "title": "Card 2"},
        ]
        results = parse_douyin_search_items(mock_raw, limit=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].video_id, "7461911073162791001")
        self.assertEqual(results[1].video_id, "7461911073162791002")

    # 12. DB duplicate detection
    def test_12_db_duplicate_detection(self):
        test_url = "https://www.douyin.com/video/7461911073162791777"
        self.db.query(Video).filter(Video.douyin_url == test_url).delete()
        self.db.commit()

        v = Video(
            video_id="V_DUPTEST",
            douyin_url=test_url,
            status="FOUND"
        )
        self.db.add(v)
        self.db.commit()

        try:
            mock_elements = [
                {"href": "/video/7461911073162791777", "title": "Already in DB"},
                {"href": "/video/7461911073162791888", "title": "Brand new video"},
            ]
            parsed = parse_douyin_search_items(mock_elements, limit=10, db=self.db)
            self.assertEqual(len(parsed), 2)
            self.assertTrue(parsed[0].already_imported)
            self.assertFalse(parsed[1].already_imported)
        finally:
            self.db.query(Video).filter(Video.douyin_url == test_url).delete()
            self.db.commit()

    # 13. Optional metadata missing
    def test_13_optional_metadata_missing(self):
        sparse_elements = [
            {"href": "/video/7461911073162791005"}
        ]
        parsed = parse_douyin_search_items(sparse_elements, limit=5)
        self.assertEqual(len(parsed), 1)
        item = parsed[0]
        self.assertEqual(item.video_id, "7461911073162791005")
        self.assertEqual(item.title, "Douyin Video 7461911073162791005")
        self.assertIsNone(item.creator)
        self.assertIsNone(item.thumbnail_url)
        self.assertIsNone(item.views)

    # 14. Requested result limit
    def test_14_requested_result_limit(self):
        many_elements = [
            {"href": f"/video/74619110731627910{i:02d}", "title": f"Video {i}"}
            for i in range(20)
        ]
        parsed_4 = parse_douyin_search_items(many_elements, limit=4)
        self.assertEqual(len(parsed_4), 4)

    # 15. User-triggered reconnect
    def test_15_user_triggered_reconnect(self):
        with patch("app.services.douyin_browser_service.douyin_browser_service.is_cdp_reachable", return_value=True):
            with patch("app.services.douyin_browser_service.douyin_browser_service.get_status", return_value={"state": STATE_CONNECTED, "connected": True}):
                res = self.client.get("/api/douyin-browser/status")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["state"], STATE_CONNECTED)

    # 16. Clean close
    def test_16_clean_close(self):
        mock_proc = MagicMock()
        self.browser_service._process = mock_proc
        with patch.object(self.browser_service, "is_cdp_reachable", return_value=False):
            res = self.browser_service.close_browser()
            self.assertTrue(res["success"])
            self.assertEqual(res["state"], STATE_NOT_CONNECTED)
            mock_proc.terminate.assert_called_once()
            self.assertIsNone(self.browser_service._process)

    # 17. Import selected compatibility
    def test_17_import_selected_compatibility(self):
        test_canon = "https://www.douyin.com/video/7461911073162791991"
        self.db.query(Video).filter(Video.douyin_url == test_canon).delete()
        self.db.commit()

        try:
            payload = {
                "videos": [
                    {
                        "canonical_url": test_canon,
                        "video_id": "7461911073162791991",
                        "title": "Nồi cơm điện mini tiện dụng",
                        "creator": "Đồ gia dụng thông minh",
                        "views": "300000"
                    }
                ],
                "product_id": None
            }
            res = self.client.post("/api/douyin-browser/import-selected", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["imported_count"], 1)
        finally:
            self.db.query(Video).filter(Video.douyin_url == test_canon).delete()
            self.db.commit()

    # 18. Existing SnapTikTok workflow untouched
    def test_18_existing_snaptiktok_workflow_untouched(self):
        downloader = DownloaderService()
        self.assertTrue(hasattr(downloader, "download_video"))
        self.assertTrue(hasattr(downloader, "download_and_attach"))
        self.assertTrue(hasattr(downloader, "attach_local_video"))


class TestDouyinConnectLoginUX(unittest.TestCase):
    """
    10-Point Regression Suite for Connect & Login UX enhancements.
    """

    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.client = TestClient(app)
        self.test_profile = BASE_DIR / "runtime" / "test_douyin_ux_profile"
        self.browser_service = DouyinBrowserService(profile_dir=self.test_profile, cdp_port=9333)

    def tearDown(self):
        self.db.close()

    # 1. Click Connect -> launches browser on port 9333
    def test_19_connect_launches_on_port_9333(self):
        self.assertEqual(self.browser_service.cdp_port, 9333)
        self.assertEqual(DEFAULT_CDP_PORT, 9333)

        launched_args = []
        def mock_popen(args, **kwargs):
            launched_args.extend(args)
            m = MagicMock()
            m.poll.return_value = None
            m.pid = 8888
            return m

        with patch("subprocess.Popen", side_effect=mock_popen):
            with patch.object(self.browser_service, "is_cdp_reachable", side_effect=[False, True, True]):
                with patch.object(self.browser_service, "get_status", return_value={"state": STATE_LOGIN_REQUIRED, "connected": True}):
                    with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://test", "url": "https://www.douyin.com/"}):
                        with patch.object(self.browser_service, "_trigger_login_modal", return_value={"opened": True}):
                            res = self.browser_service.open_browser()
                            self.assertTrue(res["success"])
                            # Verify port 9333 argument in launched command
                            self.assertTrue(any("--remote-debugging-port=9333" in arg for arg in launched_args))
                            self.assertFalse(any("--remote-debugging-port=9222" in arg for arg in launched_args))

    # 2. Opens douyin.com
    def test_20_opens_douyin_com(self):
        navigated_urls = []
        def mock_nav(ws_url, target_url, timeout=5.0):
            navigated_urls.append(target_url)
            return True

        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://test", "url": "chrome://intro/"}):
                with patch.object(self.browser_service, "_navigate_tab", side_effect=mock_nav):
                    with patch.object(self.browser_service, "get_status", return_value={"state": STATE_LOGIN_REQUIRED, "connected": True}):
                        with patch.object(self.browser_service, "_trigger_login_modal", return_value={"opened": True}):
                            res = self.browser_service.open_browser()
                            self.assertTrue(res["success"])
                            self.assertIn("https://www.douyin.com/", navigated_urls)

    # 3. Unauthenticated state -> LOGIN_REQUIRED
    def test_21_unauthenticated_state_login_required(self):
        state = detect_douyin_page_state(
            url="https://www.douyin.com/jingxuan",
            has_login_button=True,
            has_login_modal=False
        )
        self.assertEqual(state, STATE_LOGIN_REQUIRED)

        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://test", "url": "https://www.douyin.com/"}):
                with patch.object(self.browser_service, "get_status", return_value={"state": STATE_LOGIN_REQUIRED, "connected": True}):
                    with patch.object(self.browser_service, "_trigger_login_modal", return_value={"opened": True}):
                        res = self.browser_service.open_browser()
                        self.assertEqual(res["state"], STATE_LOGIN_REQUIRED)
                        self.assertIn("Vui lòng đăng nhập Douyin trong cửa sổ trình duyệt", res["message"])
                        self.assertIn("Kiểm tra lại", res["message"])

    # 4. Visible login UI open attempt
    def test_22_visible_login_ui_open_attempt(self):
        # Simulates CDP script clicking "登录" / "Login" and returning opened=True
        mock_eval_res = {"opened": True, "clicked": "登录", "tag": "DIV"}
        with patch.object(self.browser_service, "_execute_cdp_eval", return_value=mock_eval_res):
            trigger_res = self.browser_service._trigger_login_modal("ws://test")
            self.assertTrue(trigger_res["opened"])
            self.assertEqual(trigger_res["clicked"], "登录")

    # 5. Missing login button -> graceful LOGIN_REQUIRED
    def test_23_missing_login_button_graceful_login_required(self):
        # If Douyin changes DOM and login button cannot be found
        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://test", "url": "https://www.douyin.com/"}):
                with patch.object(self.browser_service, "_trigger_login_modal", return_value={"opened": False, "reason": "login_button_not_found"}):
                    with patch.object(self.browser_service, "get_status", return_value={"state": STATE_LOGIN_REQUIRED, "connected": True}):
                        res = self.browser_service.open_browser()
                        # Must NOT fail the entire connection!
                        self.assertTrue(res["success"])
                        self.assertEqual(res["state"], STATE_LOGIN_REQUIRED)
                        self.assertIn("Vui lòng đăng nhập Douyin", res["message"])

    # 6. CAPTCHA -> VERIFICATION_REQUIRED
    def test_24_captcha_verification_required(self):
        state = detect_douyin_page_state(
            url="https://www.douyin.com/search/food",
            title="验证码中间页",
            dom_text="请完成下列验证后继续"
        )
        self.assertEqual(state, STATE_VERIFICATION_REQUIRED)

        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://test", "url": "https://www.douyin.com/"}):
                with patch.object(self.browser_service, "get_status", return_value={"state": STATE_VERIFICATION_REQUIRED, "connected": True, "message": "Douyin yêu cầu xác minh."}):
                    res = self.browser_service.open_browser()
                    self.assertEqual(res["state"], STATE_VERIFICATION_REQUIRED)
                    self.assertIn("xác minh", res["message"].lower())

    # 7. Authenticated session -> CONNECTED
    def test_25_authenticated_session_connected(self):
        state = detect_douyin_page_state(
            url="https://www.douyin.com/jingxuan",
            has_avatar=True,
            has_login_button=False,
            has_login_modal=False
        )
        self.assertEqual(state, STATE_CONNECTED)

        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "_get_active_tab", return_value={"webSocketDebuggerUrl": "ws://test", "url": "https://www.douyin.com/"}):
                with patch.object(self.browser_service, "get_status", return_value={"state": STATE_CONNECTED, "connected": True}):
                    res = self.browser_service.open_browser()
                    self.assertEqual(res["state"], STATE_CONNECTED)
                    self.assertIn("sẵn sàng tìm kiếm", res["message"])

    # 8. Unrelated process on 9222 ignored
    def test_26_unrelated_process_on_9222_ignored(self):
        # Service on port 9333 must ignore WebView2 / LenovoVantage
        service_9222 = DouyinBrowserService(profile_dir=self.test_profile, cdp_port=9222)

        # Mocking 9222 answering with msedgewebview2 / LenovoVantage
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Browser": "Edg/154.0.4258.53",
            "User-Agent": "LenovoVantage/3.0.0.208"
        }

        with patch("httpx.get", return_value=mock_response):
            reachable = service_9222.is_cdp_reachable()
            self.assertFalse(reachable, "Port 9222 occupied by msedgewebview2 must be rejected and ignored!")

    # 9. No duplicate Chrome instances
    def test_27_no_duplicate_chrome_instances(self):
        with patch.object(self.browser_service, "is_cdp_reachable", return_value=True):
            with patch.object(self.browser_service, "get_status", return_value={"state": STATE_CONNECTED, "connected": True}):
                with patch("subprocess.Popen") as mock_popen:
                    res = self.browser_service.open_browser()
                    self.assertTrue(res["success"])
                    mock_popen.assert_not_called()

    # 10. Existing SnapTikTok workflow unchanged
    def test_28_existing_snaptiktok_workflow_unchanged(self):
        raw_share = "https://www.douyin.com/video/7461911073162791555"
        extracted_id = extract_douyin_video_id(raw_share)
        canonical = get_canonical_douyin_url(raw_share)
        self.assertEqual(extracted_id, "7461911073162791555")
        self.assertEqual(canonical, "https://www.douyin.com/video/7461911073162791555")

        downloader = DownloaderService()
        self.assertIsNotNone(downloader)
        self.assertTrue(callable(getattr(downloader, "download_video", None)))


if __name__ == "__main__":
    unittest.main()
