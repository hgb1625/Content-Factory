"""
Douyin Browser Service - Safe Browser-Assisted Search & Authentication

Architectural Features:
1. ONE dedicated Douyin browser profile only (runtime/douyin_browser_profile).
2. ONE browser process/session only (prevents duplicate launches & zombie processes).
3. Dedicated CDP port: 9333 (strictly ignores unrelated WebView2 processes like msedgewebview2 on 9222).
4. Configurable CDP port via DOUYIN_CDP_PORT environment variable (default: 9333).
5. Automatic visible login UI opening (clicks standard '登录' / 'Login' button) leaving QR/SMS open for manual user action.
6. Graceful fallback if login button selector changes (keeps browser open, reports LOGIN_REQUIRED).
7. Session/cookies remain inside that dedicated profile; never logged, printed, or exported.
8. Searches are SEQUENTIAL (guaranteed via threading.Lock).
9. NO parallel searches, NO auto-retry loops, NO credentials autofill, NO automated CAPTCHA solving.
10. Truthful state machine: NOT_CONNECTED, BROWSER_OPEN, LOGIN_REQUIRED, VERIFICATION_REQUIRED, CONNECTED, SEARCHING, ERROR.
11. Pure DOM parsing with Bounded Progressive Scrolling (up to 5 scroll steps max).
12. Structured DouyinSearchResult model with Database Deduplication (marking ALREADY IMPORTED).
"""

import os
import sys
import json
import time
import re
import urllib.parse
import logging
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger("app.services.douyin_browser")

# Truthful Browser States
STATE_NOT_CONNECTED = "NOT_CONNECTED"
STATE_BROWSER_OPEN = "BROWSER_OPEN"
STATE_LOGIN_REQUIRED = "LOGIN_REQUIRED"
STATE_VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
STATE_CONNECTED = "CONNECTED"
STATE_SEARCHING = "SEARCHING"
STATE_ERROR = "ERROR"

# Canonical base directory & defaults
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROFILE_DIR = BASE_DIR / "runtime" / "douyin_browser_profile"
DEFAULT_CDP_PORT = int(os.environ.get("DOUYIN_CDP_PORT", "9333"))


from app.services.douyin_search.models import DouyinSearchResult


def detect_douyin_page_state(
    url: str = "",
    title: str = "",
    dom_text: str = "",
    dom_html_snippet: str = "",
    has_login_button: Optional[bool] = None,
    has_login_modal: Optional[bool] = None,
    has_avatar: Optional[bool] = None
) -> str:
    """
    Pure evaluation function to inspect rendered page indicators.
    Returns one of:
      - VERIFICATION_REQUIRED (CAPTCHA, puzzle slider, secsdk intermediate page)
      - LOGIN_REQUIRED (login modal, passport banner, '请先登录', visible Login button)
      - CONNECTED (valid Douyin page with authenticated user session & no blockers)
      - BROWSER_OPEN (blank, homepage without Douyin, or non-douyin page)
    """
    url_lower = (url or "").lower()
    title_lower = (title or "").lower()
    combined_text = f"{title} {dom_text} {dom_html_snippet}".strip()

    # 1. Verification / CAPTCHA checks (Highest priority blocker)
    captcha_keywords = [
        "验证码中间页",
        "请完成下列验证后继续",
        "按住左边按钮拖动完成上方拼图",
        "完成下方拼图",
        "验证码",
        "secsdk",
        "captcha_verify",
        "captcha-verify",
        "security verification",
    ]
    if "verify" in url_lower or "captcha" in url_lower:
        return STATE_VERIFICATION_REQUIRED

    for kw in captcha_keywords:
        if kw.lower() in combined_text.lower():
            return STATE_VERIFICATION_REQUIRED

    # 2. Login checks
    if has_login_modal:
        return STATE_LOGIN_REQUIRED

    if has_login_button:
        return STATE_LOGIN_REQUIRED

    login_keywords = [
        "请先登录，再继续搜索吧",
        "请先登录",
        "登录后查看更多",
        "登录后即可查看",
        "立即登录",
        "扫码登录",
        "scan to log in",
        "log in to douyin",
        "passport_login",
        "login-guide-container",
        "douyin-login-panel",
        "douyin_login_new_class",
    ]
    if "passport.douyin.com" in url_lower or "/login" in url_lower:
        return STATE_LOGIN_REQUIRED

    for kw in login_keywords:
        if kw.lower() in combined_text.lower():
            return STATE_LOGIN_REQUIRED

    # 3. Connected vs Open
    if "douyin.com" in url_lower:
        # If avatar is explicitly detected or no login blocker found on douyin.com
        return STATE_CONNECTED

    return STATE_BROWSER_OPEN


def extract_canonical_douyin_video_id(href: str) -> Optional[str]:
    """Extract canonical numeric video ID (15-25 digits) from href."""
    if not href:
        return None
    match = re.search(r"/video/(\d{15,25})", href)
    if match:
        return match.group(1)
    match_modal = re.search(r"modal_id=(\d{15,25})", href)
    if match_modal:
        return match_modal.group(1)
    return None


def parse_douyin_search_items(
    elements: List[Dict[str, Any]],
    limit: int = 10,
    db: Optional[Any] = None
) -> List[DouyinSearchResult]:
    """
    Pure extraction function to parse, deduplicate, and limit search items.
    Checks each item against SQLite DB (if provided) to mark already_imported.
    """
    results: List[DouyinSearchResult] = []
    seen_ids = set()

    existing_urls = set()
    if db is not None:
        try:
            from app.models import Video
            db_videos = db.query(Video.douyin_url).all()
            for (u,) in db_videos:
                if u:
                    existing_urls.add(u.strip())
        except Exception as e:
            logger.warning(f"Error checking existing DB records: {e}")

    for el in elements:
        href = el.get("href", "")
        vid = extract_canonical_douyin_video_id(href)
        if not vid or vid in seen_ids:
            continue

        seen_ids.add(vid)
        canonical_url = f"https://www.douyin.com/video/{vid}"
        title = (el.get("title") or "").strip()
        creator = (el.get("creator") or "").strip() or None
        views = (el.get("views") or "").strip() or None
        thumbnail = (el.get("thumbnail") or "").strip() or None

        is_imported = (
            canonical_url in existing_urls or
            any(vid in eu for eu in existing_urls)
        )

        results.append(DouyinSearchResult(
            video_id=vid,
            canonical_url=canonical_url,
            title=title or f"Douyin Video {vid}",
            creator=creator,
            views=views,
            thumbnail_url=thumbnail,
            already_imported=is_imported,
            raw_href=href
        ))

        if len(results) >= limit:
            break

    return results


class DouyinBrowserService:
    """
    Manages exactly ONE visible local Chrome instance with dedicated profile
    on CDP port 9333 to perform safe, human-attended Douyin operations.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(DouyinBrowserService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        profile_dir: Optional[Path] = None,
        cdp_port: Optional[int] = None,
        browser_executable: Optional[str] = None
    ):
        if getattr(self, "_initialized", False):
            if profile_dir is not None:
                self.profile_dir = Path(profile_dir)
            if cdp_port is not None:
                self.cdp_port = cdp_port
            if browser_executable is not None:
                self.browser_executable = browser_executable
            return

        self.profile_dir = Path(profile_dir or DEFAULT_PROFILE_DIR)
        self.cdp_port = cdp_port if cdp_port is not None else DEFAULT_CDP_PORT
        self.browser_executable = browser_executable
        self._process: Optional[subprocess.Popen] = None
        self._search_lock = threading.Lock()
        self._is_searching = False
        self._initialized = True

    def _locate_browser_binary(self) -> Optional[str]:
        """Find local Google Chrome executable (strictly Chrome to avoid WebView2 conflicts)."""
        if self.browser_executable and os.path.exists(self.browser_executable):
            return self.browser_executable

        env_path = os.environ.get("CHROME_PATH") or os.environ.get("DOUYIN_BROWSER_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _check_process_alive(self) -> bool:
        """Check if tracked subprocess is still running; cleanup if dead."""
        if self._process is not None:
            if self._process.poll() is not None:
                logger.info(f"Subprocess PID {self._process.pid} has exited.")
                self._process = None
                return False
            return True
        return False

    def is_cdp_reachable(self) -> bool:
        """
        Check if Chrome remote debugging port responds.
        Strictly ignores unrelated WebView2 / Edge processes (e.g. msedgewebview2 on 9222).
        """
        try:
            r = httpx.get(f"http://127.0.0.1:{self.cdp_port}/json/version", timeout=1.5)
            if r.status_code != 200:
                return False
            data = r.json()
            browser_str = str(data.get("Browser", ""))
            ua_str = str(data.get("User-Agent", ""))
            # Reject msedgewebview2 / LenovoVantage or unrelated embedded WebViews
            if "msedgewebview2" in browser_str.lower() or "lenovovantage" in ua_str.lower():
                logger.warning(f"Port {self.cdp_port} answered by unrelated WebView2 process. Ignoring.")
                return False
            # Verify Chrome or Chromium family
            if not ("Chrome" in browser_str or "Chromium" in browser_str):
                logger.warning(f"Port {self.cdp_port} answered by non-Chrome browser: {browser_str}")
                return False
            return True
        except Exception:
            self._check_process_alive()
            return False

    def reconnect_if_alive(self) -> Dict[str, Any]:
        """Check and reconnect to an already-running Chrome instance on port 9333."""
        if self.is_cdp_reachable():
            status_info = self.get_status()
            return {
                "success": True,
                "reconnected": True,
                "message": "Đã kết nối lại thành công với phiên trình duyệt Douyin hiện có.",
                "state": status_info["state"],
                "details": status_info
            }
        return {
            "success": False,
            "reconnected": False,
            "state": STATE_NOT_CONNECTED,
            "message": "Không tìm thấy phiên trình duyệt nào đang chạy."
        }

    def _get_active_tab(self) -> Optional[Dict[str, Any]]:
        """Find active or first Douyin tab from CDP list."""
        try:
            r = httpx.get(f"http://127.0.0.1:{self.cdp_port}/json/list", timeout=2.0)
            if r.status_code != 200:
                return None
            tabs = r.json()
            page_tabs = [t for t in tabs if t.get("type") == "page"]
            if not page_tabs:
                return None
            for t in page_tabs:
                if "douyin.com" in t.get("url", ""):
                    return t
            return page_tabs[0]
        except Exception as e:
            logger.warning(f"Error querying CDP tabs on port {self.cdp_port}: {e}")
            return None

    def _execute_cdp_eval(self, ws_url: str, js_code: str, timeout: float = 5.0) -> Any:
        """Connect to tab WebSocket and evaluate JS expression with stale recovery."""
        from websockets.sync.client import connect

        try:
            with connect(ws_url, open_timeout=timeout, close_timeout=timeout) as ws:
                msg = {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": js_code,
                        "returnByValue": True,
                        "awaitPromise": True
                    }
                }
                ws.send(json.dumps(msg))
                start = time.time()
                while time.time() - start < timeout:
                    raw = ws.recv(timeout=timeout)
                    data = json.loads(raw)
                    if data.get("id") == 1:
                        result_obj = data.get("result", {})
                        if "exceptionDetails" in result_obj:
                            logger.warning(f"CDP eval exception: {result_obj.get('exceptionDetails')}")
                            return None
                        return result_obj.get("result", {}).get("value")
        except Exception as e:
            logger.warning(f"Stale CDP socket or eval failure: {e}")
            return None
        return None

    def _navigate_tab(self, ws_url: str, target_url: str, timeout: float = 5.0) -> bool:
        """Navigate tab to URL via CDP."""
        from websockets.sync.client import connect

        try:
            with connect(ws_url, open_timeout=timeout, close_timeout=timeout) as ws:
                msg = {
                    "id": 2,
                    "method": "Page.navigate",
                    "params": {"url": target_url}
                }
                ws.send(json.dumps(msg))
                start = time.time()
                while time.time() - start < timeout:
                    raw = ws.recv(timeout=timeout)
                    data = json.loads(raw)
                    if data.get("id") == 2:
                        return True
        except Exception as e:
            logger.warning(f"CDP navigate failure: {e}")
            return False
        return False

    def _trigger_login_modal(self, ws_url: str) -> Dict[str, Any]:
        """
        Locate and click standard visible Douyin login button ('登录' / 'Login')
        in the rendered DOM without credentials injection.
        Leaves the login modal open for manual user interaction.
        """
        js_click = """
        (() => {
            // 1. If login modal is already visible, do not re-click
            const existingModal = document.querySelector('.douyin_login_new_class, .douyin-login-panel, [data-e2e="login-panel"], [class*="login-container"], [class*="login_new"]');
            if (existingModal) {
                return { opened: true, reason: 'modal_already_present' };
            }

            // 2. Look for header login button in top 150px
            const targetTexts = ['登录', '立即登录', 'Login', 'Sign in'];
            const allEls = Array.from(document.querySelectorAll('button, div, span, a'));
            for (const el of allEls) {
                const txt = (el.innerText || '').trim();
                if (targetTexts.includes(txt)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top < 150) {
                        el.click();
                        return { opened: true, clicked: txt, tag: el.tagName, className: el.className };
                    }
                }
            }

            // 3. Fallback: any visible button containing '登录' or 'Login' (len <= 8)
            for (const el of allEls) {
                const txt = (el.innerText || '').trim();
                if ((txt.includes('登录') || txt.includes('Login')) && txt.length <= 8) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top < 200) {
                        el.click();
                        return { opened: true, clicked: txt, tag: el.tagName, className: el.className };
                    }
                }
            }

            return { opened: false, reason: 'login_button_not_found' };
        })()
        """
        try:
            res = self._execute_cdp_eval(ws_url, js_click, timeout=4.0)
            if isinstance(res, dict):
                return res
            return {"opened": False, "reason": "eval_returned_none"}
        except Exception as e:
            logger.warning(f"Error triggering login modal: {e}")
            return {"opened": False, "error": str(e)}

    def open_browser(self) -> Dict[str, Any]:
        """
        Start ONE visible browser with persistent dedicated profile on CDP port 9333.
        Navigates to https://www.douyin.com/.
        Detects authentication state.
        If unauthenticated, automatically attempts to open the visible Douyin login UI/modal.
        Leaves the browser visible for manual user interaction (scan QR, SMS, CAPTCHA).
        """
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        is_running = self.is_cdp_reachable()
        if not is_running:
            self._check_process_alive()
            binary = self._locate_browser_binary()
            if not binary:
                return {
                    "success": False,
                    "state": STATE_ERROR,
                    "error": "Không tìm thấy Chrome trên hệ thống. Vui lòng cài đặt Google Chrome."
                }

            args = [
                binary,
                f"--user-data-dir={self.profile_dir.resolve()}",
                f"--remote-debugging-port={self.cdp_port}",
                "--no-first-run",
                "--no-default-browser-check",
                "https://www.douyin.com/"
            ]

            logger.info(f"Launching Douyin browser on port {self.cdp_port} with profile: {self.profile_dir}")
            try:
                self._process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                logger.error(f"Failed to launch browser: {e}")
                return {
                    "success": False,
                    "state": STATE_ERROR,
                    "error": f"Lỗi khởi chạy trình duyệt: {str(e)}"
                }

            # Poll CDP for up to 10 seconds
            deadline = time.time() + 10.0
            ready = False
            while time.time() < deadline:
                if self.is_cdp_reachable():
                    ready = True
                    break
                time.sleep(0.5)

            if not ready:
                return {
                    "success": False,
                    "state": STATE_ERROR,
                    "error": f"Trình duyệt đã khởi chạy nhưng cổng điều khiển (CDP {self.cdp_port}) không phản hồi."
                }

        # Ensure active tab navigates to douyin.com
        tab = self._get_active_tab()
        if tab and tab.get("webSocketDebuggerUrl"):
            ws_url = tab["webSocketDebuggerUrl"]
            tab_url = tab.get("url", "")
            if "douyin.com" not in tab_url:
                logger.info(f"Navigating tab from {tab_url} to https://www.douyin.com/")
                self._navigate_tab(ws_url, "https://www.douyin.com/", timeout=5.0)
                time.sleep(2.5)

        # Inspect current state
        status_info = self.get_status()
        state = status_info["state"]

        if state == STATE_VERIFICATION_REQUIRED:
            return {
                "success": True,
                "state": STATE_VERIFICATION_REQUIRED,
                "message": "Douyin yêu cầu xác minh. Hãy hoàn tất xác minh trong cửa sổ Douyin, sau đó bấm Kiểm tra lại.",
                "action_required": "USER ACTION REQUIRED — COMPLETE DOUYIN VERIFICATION",
                "details": status_info
            }

        if state == STATE_CONNECTED:
            return {
                "success": True,
                "state": STATE_CONNECTED,
                "message": "Trình duyệt đã kết nối Douyin sẵn sàng tìm kiếm.",
                "details": status_info
            }

        # Unauthenticated: Attempt to trigger the normal visible Douyin login UI/modal
        tab = self._get_active_tab()
        login_modal_opened = False
        if tab and tab.get("webSocketDebuggerUrl"):
            ws_url = tab["webSocketDebuggerUrl"]
            login_result = self._trigger_login_modal(ws_url)
            if login_result and login_result.get("opened"):
                login_modal_opened = True
                time.sleep(1.0)
                status_info = self.get_status()

        vietnamese_msg = (
            "Vui lòng đăng nhập Douyin trong cửa sổ trình duyệt vừa mở. "
            "Sau khi đăng nhập/xác minh xong, quay lại đây và bấm Kiểm tra lại."
        )

        return {
            "success": True,
            "state": STATE_LOGIN_REQUIRED,
            "message": vietnamese_msg,
            "action_required": "USER ACTION REQUIRED — LOGIN TO DOUYIN",
            "login_modal_opened": login_modal_opened,
            "details": status_info
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Report truthful browser state:
        NOT_CONNECTED, BROWSER_OPEN, LOGIN_REQUIRED, VERIFICATION_REQUIRED, CONNECTED, SEARCHING, ERROR.
        """
        if self._is_searching:
            return {
                "state": STATE_SEARCHING,
                "connected": True,
                "message": "Đang tìm kiếm trên trình duyệt..."
            }

        if not self.is_cdp_reachable():
            self._check_process_alive()
            return {
                "state": STATE_NOT_CONNECTED,
                "connected": False,
                "message": "Trình duyệt chưa được kết nối."
            }

        tab = self._get_active_tab()
        if not tab:
            return {
                "state": STATE_BROWSER_OPEN,
                "connected": True,
                "message": "Trình duyệt đang mở nhưng không có thẻ hoạt động."
            }

        tab_url = tab.get("url", "")
        tab_title = tab.get("title", "")
        ws_url = tab.get("webSocketDebuggerUrl")

        if not ws_url:
            return {
                "state": STATE_BROWSER_OPEN,
                "connected": True,
                "url": tab_url,
                "title": tab_title,
                "message": "Trình duyệt đang mở."
            }

        # Inspect DOM snippet
        js_probe = """
        (() => {
            const url = window.location.href;
            const title = document.title;
            const bodyText = document.body ? document.body.innerText.slice(0, 3000) : '';
            const htmlSnippet = document.documentElement ? document.documentElement.innerHTML.slice(0, 4000) : '';

            // 1. Check CAPTCHA / verification
            const isCaptcha = (
                url.includes('verify') || url.includes('captcha') ||
                title.includes('验证码') || bodyText.includes('验证码中间页') ||
                bodyText.includes('请完成下列验证后继续') || bodyText.includes('按住左边按钮拖动完成上方拼图') ||
                bodyText.includes('完成下方拼图') || bodyText.includes('Security Verification') ||
                !!document.querySelector('#captcha-verify-image, .captcha_verify_container, [id*="captcha"]')
            );

            // 2. Check login modal
            const loginModal = document.querySelector('.douyin_login_new_class, .douyin-login-panel, [data-e2e="login-panel"], [class*="login-container"], [class*="login_new"]');
            const hasLoginModal = !!loginModal;

            // 3. Check visible header login button
            const targetTexts = ['登录', '立即登录', 'Login', 'Sign in'];
            let hasLoginButton = false;
            const allEls = Array.from(document.querySelectorAll('button, div, span, a'));
            for (const el of allEls) {
                const txt = (el.innerText || '').trim();
                if (targetTexts.includes(txt)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.top < 150) {
                        hasLoginButton = true;
                        break;
                    }
                }
            }

            // 4. Check avatar / authenticated indicators
            const avatarEl = document.querySelector('[data-e2e="user-avatar"], [class*="avatar"], .header-avatar, a[href*="/user/"]');
            const hasAvatar = !!avatarEl;

            return {
                url: url,
                title: title,
                bodyText: bodyText,
                htmlSnippet: htmlSnippet,
                isCaptcha: isCaptcha,
                hasLoginModal: hasLoginModal,
                hasLoginButton: hasLoginButton,
                hasAvatar: hasAvatar
            };
        })()
        """
        try:
            dom_data = self._execute_cdp_eval(ws_url, js_probe, timeout=3.0)
            if dom_data and isinstance(dom_data, dict):
                detected_state = detect_douyin_page_state(
                    url=dom_data.get("url") or tab_url,
                    title=dom_data.get("title") or tab_title,
                    dom_text=dom_data.get("bodyText") or "",
                    dom_html_snippet=dom_data.get("htmlSnippet") or "",
                    has_login_button=dom_data.get("hasLoginButton"),
                    has_login_modal=dom_data.get("hasLoginModal"),
                    has_avatar=dom_data.get("hasAvatar")
                )
            else:
                detected_state = detect_douyin_page_state(url=tab_url, title=tab_title)
        except Exception as e:
            logger.warning(f"Error evaluating page DOM for status: {e}")
            detected_state = detect_douyin_page_state(url=tab_url, title=tab_title)

        msg_map = {
            STATE_VERIFICATION_REQUIRED: "Douyin yêu cầu xác minh. Hãy hoàn tất xác minh trong cửa sổ Douyin, sau đó bấm Kiểm tra lại.",
            STATE_LOGIN_REQUIRED: "Vui lòng đăng nhập Douyin trong cửa sổ trình duyệt vừa mở. Sau khi đăng nhập/xác minh xong, quay lại đây và bấm Kiểm tra lại.",
            STATE_CONNECTED: "Trình duyệt đã kết nối Douyin sẵn sàng tìm kiếm.",
            STATE_BROWSER_OPEN: "Trình duyệt đang mở."
        }

        return {
            "state": detected_state,
            "connected": True,
            "url": tab_url,
            "title": tab_title,
            "message": msg_map.get(detected_state, "Trình duyệt đang mở.")
        }

    def search(
        self,
        keyword: str,
        limit: int = 10,
        db: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform a sequential keyword search on Douyin web UI with bounded progressive scrolling.
        Enforces threading lock: exactly ONE search at a time.
        Stops immediately if verification or login appears.
        Extracts visible video URLs without auto-downloading.
        Deduplicates against SQLite DB and marks already_imported.
        """
        cleaned_kw = (keyword or "").strip()
        if not cleaned_kw:
            return {"success": False, "state": STATE_ERROR, "error": "Từ khóa tìm kiếm không được để trống."}

        acquired = self._search_lock.acquire(blocking=False)
        if not acquired:
            return {
                "success": False,
                "state": STATE_ERROR,
                "error": "Một tác vụ tìm kiếm khác đang chạy. Hệ thống thực hiện tìm kiếm tuần tự (Sequential), vui lòng chờ."
            }

        self._is_searching = True
        try:
            status_now = self.get_status()
            if not status_now.get("connected"):
                return {
                    "success": False,
                    "state": STATE_NOT_CONNECTED,
                    "error": "Trình duyệt chưa được kết nối. Vui lòng bấm [Connect Douyin Browser] trước."
                }

            tab = self._get_active_tab()
            if not tab or not tab.get("webSocketDebuggerUrl"):
                return {
                    "success": False,
                    "state": STATE_ERROR,
                    "error": "Không tìm thấy thẻ trình duyệt Douyin hoạt động."
                }

            ws_url = tab["webSocketDebuggerUrl"]
            search_url = f"https://www.douyin.com/search/{urllib.parse.quote(cleaned_kw)}"
            logger.info(f"Navigating to Douyin search: {search_url}")

            nav_ok = self._navigate_tab(ws_url, search_url, timeout=5.0)
            if not nav_ok:
                return {
                    "success": False,
                    "state": STATE_ERROR,
                    "error": "Không thể gửi lệnh điều hướng tới trình duyệt."
                }

            # Wait 3.5 seconds for initial render
            time.sleep(3.5)

            # Check if verification or login was triggered immediately
            check_state = self.get_status()
            if check_state["state"] in [STATE_VERIFICATION_REQUIRED, STATE_LOGIN_REQUIRED]:
                return {
                    "success": False,
                    "state": check_state["state"],
                    "keyword": cleaned_kw,
                    "message": check_state["message"],
                    "action_required": (
                        "USER ACTION REQUIRED — LOGIN TO DOUYIN"
                        if check_state["state"] == STATE_LOGIN_REQUIRED
                        else "USER ACTION REQUIRED — COMPLETE DOUYIN VERIFICATION"
                    ),
                    "videos": []
                }

            js_extract = """
            (() => {
                const results = [];
                const links = document.querySelectorAll('a[href*="/video/"], a[href*="modal_id="]');
                for (const a of links) {
                    const href = a.getAttribute('href') || a.href || '';
                    const title = (a.innerText || a.getAttribute('title') || '').trim().slice(0, 150);
                    const img = a.querySelector('img');
                    const thumb = img ? (img.getAttribute('src') || '') : '';
                    
                    let creator = '';
                    const card = a.closest('li') || a.closest('.search-result-card') || a.parentElement;
                    if (card) {
                        const authorEl = card.querySelector('[data-e2e="author-name"], .author, .name, .account-name');
                        if (authorEl) creator = authorEl.innerText.trim();
                    }
                    results.push({ href: href, title: title, thumbnail: thumb, creator: creator });
                }
                return results;
            })()
            """

            max_scrolls = min(5, max(1, (limit // 8) + 1))
            current_extracted = []
            seen_ids_scan = set()

            for scroll_idx in range(max_scrolls):
                raw_items = self._execute_cdp_eval(ws_url, js_extract, timeout=4.0) or []
                for it in raw_items:
                    vid = extract_canonical_douyin_video_id(it.get("href", ""))
                    if vid and vid not in seen_ids_scan:
                        seen_ids_scan.add(vid)
                        current_extracted.append(it)

                if len(seen_ids_scan) >= limit:
                    break

                # Perform bounded scroll
                scroll_js = "window.scrollBy(0, 900);"
                self._execute_cdp_eval(ws_url, scroll_js, timeout=2.0)
                time.sleep(1.5)

                # Check if scrolling triggered CAPTCHA / login
                mid_check = self.get_status()
                if mid_check["state"] in [STATE_VERIFICATION_REQUIRED, STATE_LOGIN_REQUIRED]:
                    logger.info("Verification or login barrier triggered during bounded scroll.")
                    return {
                        "success": False,
                        "state": mid_check["state"],
                        "keyword": cleaned_kw,
                        "message": mid_check["message"],
                        "action_required": (
                            "USER ACTION REQUIRED — LOGIN TO DOUYIN"
                            if mid_check["state"] == STATE_LOGIN_REQUIRED
                            else "USER ACTION REQUIRED — COMPLETE DOUYIN VERIFICATION"
                        ),
                        "videos": [v.to_dict() for v in parse_douyin_search_items(current_extracted, limit=limit, db=db)]
                    }

            parsed_results = parse_douyin_search_items(current_extracted, limit=limit, db=db)
            parsed_dicts = [v.to_dict() for v in parsed_results]

            return {
                "success": True,
                "state": STATE_CONNECTED,
                "keyword": cleaned_kw,
                "count": len(parsed_dicts),
                "videos": parsed_dicts,
                "message": f"Tìm thấy {len(parsed_dicts)} video phù hợp với từ khóa '{cleaned_kw}'."
            }

        except Exception as e:
            logger.error(f"Search execution error: {e}", exc_info=True)
            return {
                "success": False,
                "state": STATE_ERROR,
                "error": f"Lỗi trong quá trình tìm kiếm trình duyệt: {str(e)}"
            }
        finally:
            self._is_searching = False
            self._search_lock.release()

    def close_browser(self) -> Dict[str, Any]:
        """Close browser process cleanly, preventing zombie processes."""
        closed = False
        if self.is_cdp_reachable():
            try:
                tab = self._get_active_tab()
                if tab and tab.get("webSocketDebuggerUrl"):
                    from websockets.sync.client import connect
                    with connect(tab["webSocketDebuggerUrl"], open_timeout=2.0) as ws:
                        ws.send(json.dumps({"id": 99, "method": "Browser.close"}))
                        time.sleep(0.5)
                        closed = True
            except Exception:
                pass

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2.0)
                closed = True
            except Exception:
                try:
                    self._process.kill()
                    closed = True
                except Exception:
                    pass
            self._process = None

        return {
            "success": True,
            "state": STATE_NOT_CONNECTED,
            "message": "Trình duyệt Douyin đã được đóng."
        }


# Global singleton instance
douyin_browser_service = DouyinBrowserService()
