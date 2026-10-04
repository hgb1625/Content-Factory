"""
Local Library Video Source Provider

Sources verified real media files from a local stock repository or directory.
Useful for offline production environments, test suites, and internal video assets.
Guarantees:
- Validates media integrity with ffprobe.
- Generates deterministic canonical IDs based on file identity.
- Refuses to report success if files do not exist or are invalid.
"""
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List

from app.services.video_source.base import VideoSourceProvider, SourceCandidate, SourceStatus
from app.services.downloader_service import validate_downloaded_media

logger = logging.getLogger("app.services.video_source.local_library")


class LocalLibrarySourceProvider(VideoSourceProvider):
    """Local Library implementation of VideoSourceProvider."""

    def __init__(self, library_dir: Optional[Path] = None):
        from app.database import ORIGINAL_DIR
        self.library_dir = library_dir or ORIGINAL_DIR
        self._consumed_paths: set = set()

    @property
    def provider_id(self) -> str:
        return "local_library"

    @property
    def display_name(self) -> str:
        return "Local Stock Library"

    def is_configured(self) -> bool:
        return self.library_dir.exists() and self.library_dir.is_dir()

    def search_source(
        self,
        query: str,
        niche: str = "",
        product_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> SourceCandidate:
        """Find an unconsumed valid local video file from the library."""
        if not self.is_configured():
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_message=f"Library directory does not exist: {self.library_dir}"
            )

        # Scan for supported video files
        candidates = sorted(
            [p for p in self.library_dir.glob("*.mp4") if p.is_file() and p.name not in self._consumed_paths],
            key=lambda p: p.name
        )

        if not candidates:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url="",
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_message=f"No unconsumed video files remaining in local library ({self.library_dir})."
            )

        chosen = candidates[0]
        self._consumed_paths.add(chosen.name)

        is_valid, meta, err = validate_downloaded_media(chosen)
        if not is_valid:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url=chosen.as_uri(),
                status=SourceStatus.SOURCE_INVALID,
                error_message=f"Local media validation failed: {err}"
            )

        canon_id = f"local:{chosen.stem}"
        return SourceCandidate(
            provider=self.provider_id,
            canonical_source_id=canon_id,
            canonical_url=chosen.as_uri(),
            download_url=str(chosen.resolve()),
            title=chosen.stem,
            duration=meta.get("duration"),
            width=meta.get("width"),
            height=meta.get("height"),
            status=SourceStatus.SOURCE_FOUND,
            metadata={"file_path": str(chosen), "product_id": product_id}
        )

    def resolve_source(self, url_or_id: str) -> SourceCandidate:
        path = Path(url_or_id)
        if not path.is_file():
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url=url_or_id,
                status=SourceStatus.SOURCE_INVALID,
                error_message=f"File not found: {url_or_id}"
            )

        is_valid, meta, err = validate_downloaded_media(path)
        if not is_valid:
            return SourceCandidate(
                provider=self.provider_id,
                canonical_source_id="",
                canonical_url=path.as_uri(),
                status=SourceStatus.SOURCE_INVALID,
                error_message=f"Local media validation failed: {err}"
            )

        return SourceCandidate(
            provider=self.provider_id,
            canonical_source_id=f"local:{path.stem}",
            canonical_url=path.as_uri(),
            download_url=str(path.resolve()),
            title=path.stem,
            duration=meta.get("duration"),
            width=meta.get("width"),
            height=meta.get("height"),
            status=SourceStatus.SOURCE_FOUND
        )

    def reset_consumed(self) -> None:
        self._consumed_paths.clear()
