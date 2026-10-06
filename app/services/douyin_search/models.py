"""
Douyin Search Models & Canonical URL Helpers
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("app.services.douyin_search.models")


@dataclass
class DouyinSearchResult:
    """Normalized search result model across all search providers."""
    video_id: str
    canonical_url: str
    title: str = ""
    creator: str = ""
    thumbnail_url: str = ""
    views: str = ""
    source: str = "browser"  # 'browser' | 'serpapi' | etc.
    already_imported: bool = False
    raw_url: str = ""
    raw_href: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("raw_url") and d.get("raw_href"):
            d["raw_url"] = d["raw_href"]
        if not d.get("raw_href") and d.get("raw_url"):
            d["raw_href"] = d["raw_url"]
        return d


def extract_canonical_douyin_video_id(url_or_href: str) -> Optional[str]:
    """
    Extract canonical numeric video ID (15-25 digits) from Douyin video URLs.
    Rejects user profiles, search pages, live rooms, and non-Douyin domains.
    Examples:
      - https://www.douyin.com/video/7461911073162791080 -> 7461911073162791080
      - https://www.douyin.com/jingxuan/search/test?modal_id=7461911073162791080 -> 7461911073162791080
      - https://www.douyin.com/user/MS4wLj... -> None
    """
    if not url_or_href:
        return None
    s = str(url_or_href).strip()

    # Match /video/<id>
    match = re.search(r"/video/(\d{15,25})", s)
    if match:
        return match.group(1)

    # Match modal_id=<id>
    match_modal = re.search(r"modal_id=(\d{15,25})", s)
    if match_modal:
        return match_modal.group(1)

    return None


def deduplicate_against_db(
    results: List[DouyinSearchResult],
    db: Optional[Any] = None
) -> List[DouyinSearchResult]:
    """
    Mark already_imported = True for items existing in Content Factory SQLite DB.
    """
    if db is None:
        return results

    try:
        from app.models import Video
        db_videos = db.query(Video.douyin_url).all()
        existing_urls = {u.strip() for (u,) in db_videos if u}
        for item in results:
            if item.canonical_url in existing_urls or any(item.video_id in eu for eu in existing_urls):
                item.already_imported = True
    except Exception as e:
        logger.warning(f"Error checking existing DB records: {e}")

    return results
