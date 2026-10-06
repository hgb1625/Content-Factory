"""
Base Douyin Search Provider Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class DouyinSearchProvider(ABC):
    """Abstract interface for all Douyin search providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Machine identifier (e.g. 'browser', 'serpapi')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name for UI dropdown."""
        pass

    @property
    @abstractmethod
    def requires_douyin_login(self) -> bool:
        """Whether this provider requires an active Douyin user session."""
        pass

    @abstractmethod
    def search(
        self,
        keyword: str,
        limit: int = 10,
        db: Optional[Any] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Execute search and return standardized dict response:
        {
            "success": bool,
            "provider": str,
            "state": str,
            "keyword": str,
            "count": int,
            "videos": List[Dict],
            "message": str,
            "error": Optional[str]
        }
        """
        pass
