"""
Pipeline Router — 30-Product Production Pipeline API & UI

Endpoints:
- POST /api/pipeline/run-batch: Trigger background production batch (defaults to 30 products)
- GET /api/pipeline/status/{batch_id}: Poll real-time batch progress, counts, and final state
- GET /api/pipeline/active: List recent and active batches
- GET /pipeline: Web UI dashboard for the 30-product production pipeline
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.pipeline_orchestrator import (
    get_pipeline_batch_manager,
    TARGET_PRODUCT_COUNT,
    PipelineState
)
from app.services.video_source import get_video_source_manager

logger = logging.getLogger("app.routes.pipeline")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(tags=["pipeline"])


class PipelineRunRequest(BaseModel):
    niche: str = Field(..., min_length=1, description="Target commercial niche")
    product_count: int = Field(default=TARGET_PRODUCT_COUNT, ge=1, le=50, description="Product count (default 30)")
    source_provider: Optional[str] = Field(default="pexels", description="Video source provider: pexels, local_library, douyin")
    fresh: bool = Field(default=False, description="Bypass niche cache and force fresh AI generation")


@router.get("/pipeline", response_class=HTMLResponse)
def get_pipeline_page(request: Request, db: Session = Depends(get_db)):
    """Render 30-Product Production Pipeline management UI."""
    source_mgr = get_video_source_manager()
    providers = source_mgr.list_providers()
    batch_mgr = get_pipeline_batch_manager()
    recent_batches = batch_mgr.list_batches(limit=10)

    return templates.TemplateResponse(
        request=request,
        name="pipeline.html",
        context={
            "active_page": "pipeline",
            "providers": providers,
            "target_product_count": TARGET_PRODUCT_COUNT,
            "recent_batches": recent_batches
        }
    )


@router.post("/api/pipeline/run-batch")
async def api_run_batch(request: Request, db: Session = Depends(get_db)):
    """
    Trigger production pipeline batch in background.
    Supports JSON payload or URL-encoded form data.
    Defaults strictly to product_count = 30.
    """
    niche = ""
    product_count = TARGET_PRODUCT_COUNT
    source_provider = "pexels"
    fresh = False

    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            niche = str(body.get("niche", "")).strip()
            raw_count = body.get("product_count")
            if raw_count is not None:
                try:
                    product_count = int(raw_count)
                except (ValueError, TypeError):
                    product_count = TARGET_PRODUCT_COUNT
            source_provider = body.get("source_provider") or "pexels"
            fresh = bool(body.get("fresh", False))
        except Exception as e:
            return JSONResponse(status_code=400, content={"success": False, "error": f"Invalid JSON payload: {e}"})
    else:
        form = await request.form()
        niche = str(form.get("niche", "")).strip()
        raw_count = form.get("product_count")
        if raw_count is not None and str(raw_count).strip():
            try:
                product_count = int(raw_count)
            except (ValueError, TypeError):
                product_count = TARGET_PRODUCT_COUNT
        source_provider = form.get("source_provider") or "pexels"
        fresh = str(form.get("fresh", "")).lower() in ("true", "1", "on")

    if not niche:
        return JSONResponse(status_code=400, content={"success": False, "error": "Ngách sản phẩm không được để trống."})

    if not (1 <= product_count <= 50):
        return JSONResponse(status_code=400, content={"success": False, "error": "Số lượng sản phẩm phải từ 1 đến 50."})

    batch_mgr = get_pipeline_batch_manager()
    result = batch_mgr.start_batch(
        niche=niche,
        target_count=product_count,
        source_provider_id=str(source_provider).strip().lower(),
        force_fresh_research=fresh
    )

    status_code = 200 if result.get("success") else 409
    return JSONResponse(content=result, status_code=status_code)


@router.get("/api/pipeline/status/{batch_id}")
def api_get_pipeline_status(batch_id: str):
    """Poll progress, counts, failure details, and completion state for a given batch."""
    batch_mgr = get_pipeline_batch_manager()
    status_data = batch_mgr.get_status(batch_id)

    if not status_data:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"Batch '{batch_id}' không tồn tại hoặc đã bị xóa."}
        )

    return JSONResponse(content={"success": True, "batch": status_data})


@router.get("/api/pipeline/active")
def api_get_active_batches():
    """List recent and active production pipeline batches."""
    batch_mgr = get_pipeline_batch_manager()
    batches = batch_mgr.list_batches(limit=20)
    return JSONResponse(content={"success": True, "batches": batches})
