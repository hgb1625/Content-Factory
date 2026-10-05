"""
Video Source Manager

Orchestrates video source providers (Douyin, Pexels, Local Library).
Guarantees:
- Manages registration and selection of legitimate video source providers.
- Coordinates batch source acquisition for products.
- Enforces strict canonical source ID uniqueness across batches.
- Distinguishes SOURCE_FOUND, SOURCE_UNAVAILABLE, SOURCE_BLOCKED, SOURCE_INVALID.
- Never fabricates fake sources or fake URLs.
"""
import os
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from app.models import Product, Video
from app.services.video_source.base import (
    VideoSourceProvider,
    SourceCandidate,
    SourceStatus
)
from app.services.video_source.providers.douyin import DouyinSourceProvider
from app.services.video_source.providers.pexels import PexelsSourceProvider
from app.services.video_source.providers.local_library import LocalLibrarySourceProvider

logger = logging.getLogger("app.services.video_source.manager")


class VideoSourceManager:
    """Central manager for video source acquisition providers."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self._providers: Dict[str, VideoSourceProvider] = {}
        # Register default providers
        self.register_provider(DouyinSourceProvider())
        self.register_provider(PexelsSourceProvider())
        self.register_provider(LocalLibrarySourceProvider())

    def register_provider(self, provider: VideoSourceProvider) -> None:
        if not isinstance(provider, VideoSourceProvider):
            raise TypeError("Provider must implement VideoSourceProvider.")
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: Optional[str] = None) -> VideoSourceProvider:
        pid = (provider_id or "pexels").strip().lower()
        if pid not in self._providers:
            raise ValueError(f"Video source provider '{pid}' is not registered.")
        return self._providers[pid]

    def list_providers(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        import inspect
        results = []
        for p in self._providers.values():
            try:
                sig = inspect.signature(p.is_configured)
                accepts_db = "db" in sig.parameters or any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
                is_cfg = p.is_configured(db=db) if accepts_db else p.is_configured()
            except Exception:
                is_cfg = p.is_configured()
            results.append({
                "provider_id": p.provider_id,
                "display_name": p.display_name,
                "is_configured": is_cfg
            })
        return results

    def acquire_batch_sources(
        self,
        products: List[Product],
        provider_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Acquire unique video sources for each product in the batch.
        Enforces:
        - 1:1 mapping between product and source candidate.
        - Strict deduplication: No two products may share the same canonical_source_id or canonical_url.
        - Fails honestly if any product cannot acquire a valid source.
        """
        import inspect
        provider = self.get_provider(provider_id)
        candidates: List[SourceCandidate] = []
        seen_source_ids = set()
        seen_urls = set()
        failed_items = []

        logger.info(f"Acquiring video sources for {len(products)} products via {provider.display_name}...")

        # Existing canonical source IDs in DB to prevent cross-batch duplicates
        existing_db_source_ids = set()
        existing_db_urls = set()
        if db:
            existing_videos = db.query(Video.canonical_source_id, Video.douyin_url).all()
            for csid, url in existing_videos:
                if csid:
                    existing_db_source_ids.add(csid)
                if url:
                    existing_db_urls.add(url.strip())

        sig = inspect.signature(provider.search_source)
        accepts_db = "db" in sig.parameters or any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
        opts = {"db": db}

        for idx, prod in enumerate(products, 1):
            query = prod.douyin_keywords or prod.name_vietnamese
            extra_kwargs = {"options": opts}
            if accepts_db:
                extra_kwargs["db"] = db

            cand = provider.search_source(
                query=query,
                niche=prod.niche,
                product_id=prod.product_id,
                **extra_kwargs
            )

            if not cand.is_usable:
                failed_items.append({
                    "product_id": prod.product_id,
                    "query": query,
                    "status": cand.status.value if isinstance(cand.status, SourceStatus) else str(cand.status),
                    "error": cand.error_message or "Source unusable or unavailable"
                })
                continue

            # Check intra-batch duplicate
            if cand.canonical_source_id in seen_source_ids or cand.canonical_source_id in existing_db_source_ids:
                failed_items.append({
                    "product_id": prod.product_id,
                    "canonical_source_id": cand.canonical_source_id,
                    "status": "DUPLICATE_SOURCE_ID",
                    "error": f"Duplicate canonical source ID detected: {cand.canonical_source_id}"
                })
                continue

            if cand.canonical_url in seen_urls or cand.canonical_url in existing_db_urls:
                failed_items.append({
                    "product_id": prod.product_id,
                    "canonical_url": cand.canonical_url,
                    "status": "DUPLICATE_SOURCE_URL",
                    "error": f"Duplicate canonical source URL detected: {cand.canonical_url}"
                })
                continue

            seen_source_ids.add(cand.canonical_source_id)
            seen_urls.add(cand.canonical_url)
            candidates.append(cand)

        all_succeeded = (len(candidates) == len(products)) and not failed_items
        return {
            "success": all_succeeded,
            "provider": provider.provider_id,
            "total_requested": len(products),
            "acquired_count": len(candidates),
            "candidates": candidates,
            "failed_items": failed_items
        }
