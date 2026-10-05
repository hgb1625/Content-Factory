"""
Video Source Provider Base Abstraction

Defines contracts for video source discovery and resolution across multiple providers
(Douyin, Pexels, Local Library, etc.). Distinguishes explicit source statuses:
SOURCE_FOUND, SOURCE_UNAVAILABLE, SOURCE_BLOCKED, SOURCE_INVALID.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class SourceStatus(str, Enum):
    SOURCE_FOUND = "SOURCE_FOUND"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    SOURCE_INVALID = "SOURCE_INVALID"


@dataclass
class SourceCandidate:
    """Canonical video source candidate representation."""
    provider: str
    canonical_source_id: str
    canonical_url: str
    download_url: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    product_id: Optional[str] = None
    status: SourceStatus = SourceStatus.SOURCE_FOUND
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.status == SourceStatus.SOURCE_FOUND and bool(self.download_url or self.canonical_url)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "canonical_source_id": self.canonical_source_id,
            "canonical_url": self.canonical_url,
            "download_url": self.download_url,
            "title": self.title,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "status": self.status.value if isinstance(self.status, SourceStatus) else str(self.status),
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class VideoSourceProvider(ABC):
    """Abstract Base Class for Video Source Providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for the source provider (e.g. 'douyin', 'pexels', 'local_library')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name."""
        pass

    @abstractmethod
    def is_configured(self, db: Optional[Any] = None) -> bool:
        """Check whether the provider has required credentials or environment configuration."""
        pass

    @abstractmethod
    def search_source(
        self,
        query: str,
        niche: str = "",
        product_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        db: Optional[Any] = None,
        **kwargs
    ) -> SourceCandidate:
        """
        Search for a video source matching the query or product.
        Must return explicit status:
        - SOURCE_FOUND if a legitimate, non-fake source is found.
        - SOURCE_BLOCKED if automated access is restricted (e.g. anti-bot/CAPTCHA).
        - SOURCE_UNAVAILABLE if provider is unconfigured, rate-limited, or query yielded no results.
        - SOURCE_INVALID if query/contract is malformed.
        """
        pass

    @abstractmethod
    def resolve_source(self, url_or_id: str, db: Optional[Any] = None) -> SourceCandidate:
        """Resolve a specific URL or identifier into a canonical SourceCandidate."""
        pass
