"""
SerpApi Douyin Search Provider - Zero Douyin Login Required
Discovers canonical Douyin video URLs from Google index via SerpApi.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
import httpx

from app.services.douyin_search.base import DouyinSearchProvider
from app.services.douyin_search.models import (
    DouyinSearchResult,
    extract_canonical_douyin_video_id,
    deduplicate_against_db,
)

logger = logging.getLogger("app.services.douyin_search.serpapi")

SERPAPI_SEARCH_URL = "https://serpapi.com/search"
DEFAULT_CACHE_TTL_SECONDS = 1200  # 20 minutes


class SerpApiSearchProvider(DouyinSearchProvider):
    """
    Searches Google Index through SerpApi for canonical Douyin video URLs.
    Guarantees:
      - 0 Douyin login/credentials required
      - Maximum 1 HTTP request per user search (no automatic pagination)
      - In-memory deterministic caching to prevent quota waste
      - Strict URL validation & database deduplication
      - Zero secret exposure
    """

    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        self._custom_api_key = api_key
        self._cache_ttl = cache_ttl
        # In-memory cache: keyword.lower() -> (timestamp, List[DouyinSearchResult])
        self._cache: Dict[str, tuple[float, List[DouyinSearchResult]]] = {}

    @property
    def provider_id(self) -> str:
        return "serpapi"

    @property
    def display_name(self) -> str:
        return "No Login — Google Index"

    @property
    def requires_douyin_login(self) -> bool:
        return False

    def _get_api_key(self, db: Optional[Any] = None) -> str:
        """Retrieve key from settings or environment. Never hardcoded."""
        if self._custom_api_key:
            return self._custom_api_key.strip()
        try:
            from app.config import get_serpapi_api_key
            return get_serpapi_api_key(db=db).strip()
        except Exception:
            return os.getenv("SERPAPI_API_KEY", "").strip()

    def _sanitize_error(self, err_text: str, key: str) -> str:
        """Sanitize any error message to strictly redact API key."""
        if not err_text:
            return ""
        s = str(err_text)
        if key and key in s:
            s = s.replace(key, "[REDACTED]")
        return s

    def clear_cache(self, keyword: Optional[str] = None) -> None:
        """Clear cache for specific keyword or entire cache."""
        if keyword:
            self._cache.pop(keyword.strip().lower(), None)
        else:
            self._cache.clear()

    def search(
        self,
        keyword: str,
        limit: int = 10,
        db: Optional[Any] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        cleaned_kw = (keyword or "").strip()
        if not cleaned_kw:
            return {
                "success": False,
                "provider": self.provider_id,
                "state": "ERROR",
                "keyword": "",
                "count": 0,
                "videos": [],
                "error": "Từ khóa tìm kiếm không được để trống.",
                "message": "Từ khóa tìm kiếm không được để trống."
            }

        cache_key = cleaned_kw.lower()

        # Check Cache if not force_refresh
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_items = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.info(f"SerpApi cache hit for keyword: '{cleaned_kw}'")
                # Create a fresh copy of items and re-check against DB
                results_copy = [
                    DouyinSearchResult(**item.to_dict())
                    for item in cached_items[:limit]
                ]
                deduplicate_against_db(results_copy, db)
                dicts = [r.to_dict() for r in results_copy]
                return {
                    "success": True,
                    "provider": self.provider_id,
                    "state": "CONNECTED" if dicts else "NO_RESULTS",
                    "keyword": cleaned_kw,
                    "count": len(dicts),
                    "videos": dicts,
                    "cached": True,
                    "message": (
                        f"Tìm thấy {len(dicts)} video Douyin (từ bộ nhớ đệm)."
                        if dicts else "Không tìm thấy video Douyin phù hợp trong chỉ mục web."
                    )
                }

        # Check API Key
        api_key = self._get_api_key(db=db)
        if not api_key:
            return {
                "success": False,
                "provider": self.provider_id,
                "state": "NOT_CONFIGURED",
                "keyword": cleaned_kw,
                "count": 0,
                "videos": [],
                "error": "Chưa cấu hình SerpApi. Thêm SERPAPI_API_KEY để sử dụng tìm kiếm Douyin không cần đăng nhập.",
                "message": "Chưa cấu hình SerpApi. Thêm SERPAPI_API_KEY để sử dụng tìm kiếm Douyin không cần đăng nhập."
            }

        # Build Query: site:douyin.com/video <keyword>
        dork_query = f"site:douyin.com/video {cleaned_kw}"
        params = {
            "engine": "google",
            "q": dork_query,
            "api_key": api_key,
            "num": min(max(limit * 2, 10), 20),
            "hl": "zh-cn",
            "gl": "cn",
            "output": "json"
        }

        # Execute EXACTLY ONE HTTP request
        logger.info(f"Executing single SerpApi Google Search query for '{cleaned_kw}'")
        try:
            with httpx.Client(timeout=12.0) as client:
                resp = client.get(SERPAPI_SEARCH_URL, params=params)

            # Check HTTP Status / Error conditions
            if resp.status_code in (401, 403):
                return {
                    "success": False,
                    "provider": self.provider_id,
                    "state": "INVALID_KEY",
                    "keyword": cleaned_kw,
                    "count": 0,
                    "videos": [],
                    "error": "Khóa SERPAPI_API_KEY không hợp lệ hoặc đã hết hạn.",
                    "message": "Khóa SERPAPI_API_KEY không hợp lệ hoặc đã hết hạn."
                }

            if resp.status_code == 429:
                return {
                    "success": False,
                    "provider": self.provider_id,
                    "state": "QUOTA_EXCEEDED",
                    "keyword": cleaned_kw,
                    "count": 0,
                    "videos": [],
                    "error": "Đã hết quota tìm kiếm SerpApi hiện tại.",
                    "message": "Đã hết quota tìm kiếm SerpApi hiện tại."
                }

            if resp.status_code != 200:
                raw_err = self._sanitize_error(resp.text[:200], api_key)
                return {
                    "success": False,
                    "provider": self.provider_id,
                    "state": "PROVIDER_ERROR",
                    "keyword": cleaned_kw,
                    "count": 0,
                    "videos": [],
                    "error": f"Lỗi từ dịch vụ SerpApi (HTTP {resp.status_code}): {raw_err}",
                    "message": f"Lỗi từ dịch vụ SerpApi (HTTP {resp.status_code})."
                }

            data = resp.json()

            # Check JSON payload error messages from SerpApi
            if "error" in data:
                err_msg = str(data["error"])
                sanitized = self._sanitize_error(err_msg, api_key)
                if "invalid" in err_msg.lower() and "key" in err_msg.lower():
                    return {
                        "success": False,
                        "provider": self.provider_id,
                        "state": "INVALID_KEY",
                        "keyword": cleaned_kw,
                        "count": 0,
                        "videos": [],
                        "error": "Khóa SERPAPI_API_KEY không hợp lệ.",
                        "message": "Khóa SERPAPI_API_KEY không hợp lệ."
                    }
                if "run out" in err_msg.lower() or "searches" in err_msg.lower() or "quota" in err_msg.lower():
                    return {
                        "success": False,
                        "provider": self.provider_id,
                        "state": "QUOTA_EXCEEDED",
                        "keyword": cleaned_kw,
                        "count": 0,
                        "videos": [],
                        "error": "Đã hết quota tìm kiếm SerpApi hiện tại.",
                        "message": "Đã hết quota tìm kiếm SerpApi hiện tại."
                    }
                return {
                    "success": False,
                    "provider": self.provider_id,
                    "state": "PROVIDER_ERROR",
                    "keyword": cleaned_kw,
                    "count": 0,
                    "videos": [],
                    "error": f"Lỗi SerpApi: {sanitized}",
                    "message": f"Lỗi SerpApi: {sanitized}"
                }

            # Parse organic results
            organic_results = data.get("organic_results", [])
            extracted_items: List[DouyinSearchResult] = []
            seen_ids = set()

            for item in organic_results:
                link = item.get("link", "")
                vid = extract_canonical_douyin_video_id(link)
                # Strictly ignore non-Douyin URLs or non-video pages
                if not vid or vid in seen_ids:
                    continue

                seen_ids.add(vid)
                canonical_url = f"https://www.douyin.com/video/{vid}"
                title = (item.get("title") or "").strip()
                # Clean up title if it contains suffix like "- 抖音"
                if " - 抖音" in title:
                    title = title.replace(" - 抖音", "").strip()

                thumbnail = item.get("thumbnail") or ""
                # Also check rich_snippet or thumbnail extension
                if not thumbnail and isinstance(item.get("rich_snippet"), dict):
                    extensions = item.get("rich_snippet", {}).get("bottom", {}).get("extensions", [])
                    for ext in extensions:
                        if isinstance(ext, str) and ext.startswith("http"):
                            thumbnail = ext
                            break

                extracted_items.append(DouyinSearchResult(
                    video_id=vid,
                    canonical_url=canonical_url,
                    title=title or f"Douyin Video {vid}",
                    creator="",  # Leave blank if not reliably reported in Google SERP
                    thumbnail_url=thumbnail,
                    views="",
                    source=self.provider_id,
                    already_imported=False,
                    raw_url=link
                ))

                if len(extracted_items) >= limit:
                    break

            # Update cache with un-filtered copies
            self._cache[cache_key] = (time.time(), list(extracted_items))

            # Deduplicate against DB
            deduplicate_against_db(extracted_items, db)
            dict_results = [r.to_dict() for r in extracted_items]

            if not dict_results:
                return {
                    "success": True,
                    "provider": self.provider_id,
                    "state": "NO_RESULTS",
                    "keyword": cleaned_kw,
                    "count": 0,
                    "videos": [],
                    "message": "Không tìm thấy video Douyin phù hợp trong chỉ mục web."
                }

            return {
                "success": True,
                "provider": self.provider_id,
                "state": "CONNECTED",
                "keyword": cleaned_kw,
                "count": len(dict_results),
                "videos": dict_results,
                "message": f"Tìm thấy {len(dict_results)} video Douyin qua chỉ mục web (Không cần đăng nhập)."
            }

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(f"SerpApi network error: {e}")
            return {
                "success": False,
                "provider": self.provider_id,
                "state": "NETWORK_ERROR",
                "keyword": cleaned_kw,
                "count": 0,
                "videos": [],
                "error": "Lỗi kết nối tới máy chủ tìm kiếm SerpApi.",
                "message": "Lỗi kết nối tới máy chủ tìm kiếm SerpApi."
            }
        except Exception as e:
            sanitized = self._sanitize_error(str(e), api_key)
            logger.error(f"Unexpected error in SerpApi search: {sanitized}")
            return {
                "success": False,
                "provider": self.provider_id,
                "state": "PROVIDER_ERROR",
                "keyword": cleaned_kw,
                "count": 0,
                "videos": [],
                "error": f"Lỗi không xác định: {sanitized}",
                "message": "Lỗi không xác định trong quá trình tìm kiếm."
            }
