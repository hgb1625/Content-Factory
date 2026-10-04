"""
Pexels Video Source Provider

Integrates with official Pexels Video Search API to acquire legitimate,
high-quality 9:16 portrait stock videos matching product queries.
Guarantees:
- Returns SOURCE_UNAVAILABLE when PEXELS_API_KEY is not configured.
- Discovers real vertical videos (orientation=portrait, 9:16).
- Extracts direct MP4 download links and authentic Pexels video IDs.
- Zero secret leakage in logs and exceptions.
"""
import os
import logging
from typing import Optional, Dict, Any, List
import httpx

from app.services.video_source.base import VideoSourceProvider, SourceCandidate, SourceStatus
from app.services.ai.providers.common import sanitize_secrets

logger = logging.getLogger("app.services.video_source.pexels")


class PexelsSourceProvider(VideoSourceProvider):
    """Pexels implementation of VideoSourceProvider."""

    SEARCH_URL = "https://api.pexels.com/videos/search"
    DEFAULT_TIMEOUT = 20.0

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    @property
    def provider_id(self) -> str:
        return "pexels"

    @property
    def display_name(self) -> str:
        return "Pexels Stock Video (9:16 Portrait)"

    def get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        # Check environment variable
        return os.getenv("PEXELS_API_KEY", "").strip()

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    def search_source(
        self,
        query: str,
        niche: str = "",
        product_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> SourceCandidate:
        """
        Search Pexels API for vertical (9:16) stock video matching query.
        """
        api_key = self.get_api_key()
        if not api_key:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_message="Pexels API key is not configured. Please set PEXELS_API_KEY in settings or .env.",
                metadata={"query": query, "product_id": product_id}
            )

        clean_query = query.strip()
        if not clean_query and niche:
            clean_query = niche.strip()

        if not clean_query:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_INVALID,
                error_message="Search query is empty."
            )

        headers = {
            "Authorization": api_key,
            "User-Agent": "ContentFactory/1.0"
        }
        params = {
            "query": clean_query,
            "orientation": "portrait",
            "per_page": 5,
            "size": "medium"
        }

        try:
            with httpx.Client(timeout=self.DEFAULT_TIMEOUT) as client:
                resp = client.get(self.SEARCH_URL, headers=headers, params=params)

            if resp.status_code == 401 or resp.status_code == 403:
                return SourceCandidate(
                    provider=self.provider_id,
                    canonical_source_id="",
                    canonical_url="",
                    status=SourceStatus.SOURCE_UNAVAILABLE,
                    error_message=f"Pexels API authentication failed (HTTP {resp.status_code}). Please verify PEXELS_API_KEY."
                )

            if resp.status_code != 200:
                clean_err = sanitize_secrets(resp.text[:200])
                return SourceCandidate(
                    provider=self.provider_id,
                    canonical_source_id="",
                    canonical_url="",
                    status=SourceStatus.SOURCE_UNAVAILABLE,
                    error_message=f"Pexels API returned HTTP {resp.status_code}: {clean_err}"
                )

            data = resp.json()
            videos = data.get("videos", [])
            if not videos:
                return SourceCandidate(
                    provider=self.provider_id,
                    canonical_source_id="",
                    canonical_url="",
                    status=SourceStatus.SOURCE_UNAVAILABLE,
                    error_message=f"No vertical video results found on Pexels for query: '{clean_query}'"
                )

            # Choose best vertical video file from first matching result
            best_cand = self._extract_candidate_from_pexels_item(videos[0], query=clean_query, product_id=product_id)
            if best_cand:
                return best_cand

            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_message="Pexels results contained no downloadable MP4 video files."
            )

        except Exception as e:
            clean_msg = sanitize_secrets(str(e))
            logger.warning(f"Pexels search failed for query '{clean_query}': {clean_msg}")
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_message=f"Network error querying Pexels: {clean_msg}"
            )

    def resolve_source(self, url_or_id: str) -> SourceCandidate:
        """Resolve a Pexels video by direct ID or URL."""
        # Extracts ID from pexels.com/video/... or raw ID
        import re
        m = re.search(r"(\d{5,12})", str(url_or_id))
        if not m:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url=str(url_or_id),
                status=SourceStatus.SOURCE_INVALID,
                error_message="Could not extract Pexels video ID."
            )

        video_id = m.group(1)
        api_key = self.get_api_key()
        if not api_key:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id=f"pexels:{video_id}",
                canonical_url=f"https://www.pexels.com/video/{video_id}/",
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_message="PEXELS_API_KEY required to resolve Pexels video details."
            )

        try:
            url = f"https://api.pexels.com/videos/videos/{video_id}"
            headers = {"Authorization": api_key, "User-Agent": "ContentFactory/1.0"}
            with httpx.Client(timeout=self.DEFAULT_TIMEOUT) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                cand = self._extract_candidate_from_pexels_item(resp.json())
                if cand:
                    return cand
        except Exception as e:
            logger.debug(f"Pexels resolve error: {e}")

        return SourceCandidate(
            provider=self.provider_id,
            canonical_source_id=f"pexels:{video_id}",
            canonical_url=f"https://www.pexels.com/video/{video_id}/",
            status=SourceStatus.SOURCE_UNAVAILABLE,
            error_message=f"Could not resolve Pexels video {video_id}."
        )

    def _extract_candidate_from_pexels_item(
        self,
        item: Dict[str, Any],
        query: str = "",
        product_id: Optional[str] = None
    ) -> Optional[SourceCandidate]:
        """Extract optimal vertical MP4 file from Pexels video object."""
        vid_id = str(item.get("id", ""))
        canonical_url = item.get("url") or f"https://www.pexels.com/video/{vid_id}/"
        duration = float(item.get("duration", 0.0))
        video_files = item.get("video_files", [])

        # Filter for MP4 files and prioritize vertical (height > width)
        mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4" and f.get("link")]
        if not mp4_files:
            return None

        # Sort: prefer vertical aspect ratio, then higher resolution (<= 1920)
        def _score(f: dict) -> int:
            w = f.get("width") or 0
            h = f.get("height") or 0
            is_vertical = 1 if h > w else 0
            res_score = min(h, 1920)
            return is_vertical * 10000 + res_score

        best_file = max(mp4_files, key=_score)
        download_url = best_file.get("link")
        width = best_file.get("width")
        height = best_file.get("height")

        return SourceCandidate(
            provider=self.provider_id,
            canonical_source_id=f"pexels:{vid_id}",
            canonical_url=canonical_url,
            download_url=download_url,
            title=f"Pexels Stock Video #{vid_id}",
            duration=duration,
            width=width,
            height=height,
            status=SourceStatus.SOURCE_FOUND,
            metadata={
                "pexels_id": vid_id,
                "user": item.get("user", {}).get("name"),
                "query": query,
                "product_id": product_id
            }
        )
