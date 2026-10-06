"""
Douyin Multi-Search Provider Package
"""

from typing import Dict, Any, List, Optional
from app.services.douyin_search.base import DouyinSearchProvider
from app.services.douyin_search.models import DouyinSearchResult
from app.services.douyin_search.serpapi_provider import SerpApiSearchProvider
from app.services.douyin_search.browser_provider import BrowserSearchProvider


class DouyinSearchManager:
    """Registry and dispatcher for all Douyin search providers."""

    def __init__(self):
        self._providers: Dict[str, DouyinSearchProvider] = {
            "serpapi": SerpApiSearchProvider(),
            "browser": BrowserSearchProvider(),
        }

    def register_provider(self, provider: DouyinSearchProvider) -> None:
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: Optional[str] = None) -> DouyinSearchProvider:
        pid = (provider_id or "serpapi").lower().strip()
        if pid not in self._providers:
            # Default to serpapi if invalid provider requested
            return self._providers["serpapi"]
        return self._providers[pid]

    def list_providers(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": p.provider_id,
                "name": p.display_name,
                "requires_douyin_login": p.requires_douyin_login
            }
            for p in self._providers.values()
        ]

    def search(
        self,
        keyword: str,
        limit: int = 10,
        provider_id: Optional[str] = "serpapi",
        db: Optional[Any] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        provider = self.get_provider(provider_id)
        return provider.search(keyword=keyword, limit=limit, db=db, force_refresh=force_refresh)


douyin_search_manager = DouyinSearchManager()

__all__ = [
    "DouyinSearchProvider",
    "DouyinSearchResult",
    "SerpApiSearchProvider",
    "BrowserSearchProvider",
    "DouyinSearchManager",
    "douyin_search_manager",
]
