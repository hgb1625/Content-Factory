"""
Browser-Assisted Douyin Search Provider
Wraps existing dedicated Chrome browser profile implementation.
"""

from typing import Dict, Any, Optional
from app.services.douyin_search.base import DouyinSearchProvider


class BrowserSearchProvider(DouyinSearchProvider):
    """
    Wraps existing DouyinBrowserService for live, user-attended Douyin searches.
    Requires an authenticated Douyin session in the dedicated Chrome profile.
    """

    @property
    def provider_id(self) -> str:
        return "browser"

    @property
    def display_name(self) -> str:
        return "Douyin Browser — Live"

    @property
    def requires_douyin_login(self) -> bool:
        return True

    def search(
        self,
        keyword: str,
        limit: int = 10,
        db: Optional[Any] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        from app.services.douyin_browser_service import douyin_browser_service
        res = douyin_browser_service.search(keyword=keyword, limit=limit, db=db)
        res["provider"] = self.provider_id
        if res.get("videos"):
            for v in res["videos"]:
                v["source"] = self.provider_id
        return res
