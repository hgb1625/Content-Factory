"""
Douyin Video Source Provider

Wraps DouyinService to participate in the unified VideoSourceProvider contract.
Truthfully communicates that automated discovery without an authenticated session
is BLOCKED_EXTERNAL by Douyin's dynamic token protection, while providing full
URL normalization, canonical ID extraction, and duplicate detection for resolved URLs.
"""
import logging
from typing import Optional, Dict, Any

from app.services.video_source.base import VideoSourceProvider, SourceCandidate, SourceStatus
from app.services.douyin_service import (
    normalize_douyin_url,
    extract_douyin_video_id,
    get_canonical_douyin_url,
)

logger = logging.getLogger("app.services.video_source.douyin")


class DouyinSourceProvider(VideoSourceProvider):
    """Douyin implementation of VideoSourceProvider."""

    @property
    def provider_id(self) -> str:
        return "douyin"

    @property
    def display_name(self) -> str:
        return "Douyin (TikTok China)"

    def is_configured(self, db: Optional[Any] = None) -> bool:
        # Douyin does not require an API key, but automated search requires browser session
        return True

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
        Douyin public search without authenticated session and signing tokens
        is restricted by Douyin bot protection. Returns SOURCE_BLOCKED.
        """
        logger.info(f"Douyin automated discovery requested for query: '{query}'")
        return SourceCandidate(
            provider=self.provider_id,
            canonical_source_id="",
            canonical_url="",
            status=SourceStatus.SOURCE_BLOCKED,
            error_message=(
                "Douyin automated public search is restricted by platform bot protection. "
                "Manual URL import or alternative video source provider required."
            ),
            metadata={"query": query, "product_id": product_id}
        )

    def resolve_source(self, url_or_id: str) -> SourceCandidate:
        """Resolve a Douyin URL into a canonical SourceCandidate."""
        norm_url = normalize_douyin_url(url_or_id)
        if not norm_url:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_INVALID,
                error_message="Empty or invalid Douyin URL provided."
            )

        vid_id = extract_douyin_video_id(norm_url)
        canon_url = get_canonical_douyin_url(norm_url)

        if not vid_id:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url=canon_url,
                status=SourceStatus.SOURCE_INVALID,
                error_message="Could not extract Douyin video ID from URL."
            )

        return SourceCandidate(
            provider=self.provider_id,
            canonical_source_id=f"douyin:{vid_id}",
            canonical_url=canon_url,
            status=SourceStatus.SOURCE_FOUND,
            metadata={"raw_url": norm_url, "modal_id": vid_id}
        )
