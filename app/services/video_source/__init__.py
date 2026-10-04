"""
Video Source Package

Provides unified provider abstraction for legitimate video source acquisition,
resolution, and deduplication across Douyin, Pexels, and local sources.
"""
from app.services.video_source.base import (
    VideoSourceProvider,
    SourceCandidate,
    SourceStatus,
)
from app.services.video_source.manager import VideoSourceManager


def get_video_source_manager() -> VideoSourceManager:
    """Get the singleton instance of VideoSourceManager."""
    return VideoSourceManager()


__all__ = [
    "VideoSourceProvider",
    "SourceCandidate",
    "SourceStatus",
    "VideoSourceManager",
    "get_video_source_manager",
]
