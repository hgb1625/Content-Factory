from fastapi import APIRouter, Depends, Request, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel

from app.database import get_db
from app.models import Product, Video
from app.services.douyin_service import DouyinService

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["videos"])
douyin_service = DouyinService()


class ManualImportRequest(BaseModel):
    douyin_url: str
    product_id: Optional[str] = None
    views: Optional[str] = None
    thumbnail: Optional[str] = None


class BulkActionRequest(BaseModel):
    video_ids: List[str]


@router.get("/products", response_class=HTMLResponse)
def get_products_page(request: Request, db: Session = Depends(get_db)):
    products = db.query(Product).order_by(Product.id.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="products.html",
        context={
            "products": products,
            "active_page": "products"
        }
    )


@router.get("/videos", response_class=HTMLResponse)
def get_videos_page(
    request: Request,
    sort: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    sort_by_views = (sort == "views")
    videos = douyin_service.get_sorted_videos(db, status_filter=status, sort_by_views=sort_by_views)
    return templates.TemplateResponse(
        request=request,
        name="videos.html",
        context={
            "videos": videos,
            "sort": sort,
            "status_filter": status,
            "active_page": "videos"
        }
    )


@router.get("/review", response_class=HTMLResponse)
def get_review_page(
    request: Request,
    sort: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    sort_by_views = (sort == "views")
    videos = douyin_service.get_sorted_videos(db, status_filter="FOUND", sort_by_views=sort_by_views)
    products = db.query(Product).order_by(Product.id.desc()).all()

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "videos": videos,
            "products": products,
            "sort": sort,
            "active_page": "review",
            "message": None,
            "error": None
        }
    )


@router.post("/videos/manual-import", response_class=HTMLResponse)
def manual_import_video_form(
    request: Request,
    douyin_url: str = Form(...),
    product_id: Optional[str] = Form(None),
    views: Optional[str] = Form(None),
    thumbnail: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    result = douyin_service.add_video(
        db=db,
        douyin_url=douyin_url,
        product_id=product_id if product_id and product_id.strip() else None,
        views=views.strip() if views else None,
        thumbnail=thumbnail.strip() if thumbnail else None
    )

    videos = douyin_service.get_sorted_videos(db, status_filter="FOUND")
    products = db.query(Product).order_by(Product.id.desc()).all()

    if result["success"]:
        message = result["message"]
        error = None
    else:
        message = None
        error = result["error"]

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "videos": videos,
            "products": products,
            "active_page": "review",
            "message": message,
            "error": error
        },
        status_code=200 if result["success"] else 400
    )


@router.post("/api/videos/manual-import")
def api_manual_import(payload: ManualImportRequest, db: Session = Depends(get_db)):
    result = douyin_service.add_video(
        db=db,
        douyin_url=payload.douyin_url,
        product_id=payload.product_id,
        views=payload.views,
        thumbnail=payload.thumbnail
    )
    return JSONResponse(content=result, status_code=200 if result["success"] else 400)


@router.post("/api/videos/approve")
def api_approve_video(payload: dict, db: Session = Depends(get_db)):
    video_id = payload.get("video_id")
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        return JSONResponse(status_code=404, content={"success": False, "error": f"Không tìm thấy video {video_id}"})

    video.approved = True
    video.status = "APPROVED"
    db.commit()
    return {"success": True, "video_id": video_id, "status": "APPROVED"}


@router.post("/api/videos/reject")
def api_reject_video(payload: dict, db: Session = Depends(get_db)):
    video_id = payload.get("video_id")
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        return JSONResponse(status_code=404, content={"success": False, "error": f"Không tìm thấy video {video_id}"})

    video.status = "REJECTED"
    db.commit()
    return {"success": True, "video_id": video_id, "status": "REJECTED"}


@router.post("/api/videos/bulk-approve")
def api_bulk_approve(payload: BulkActionRequest, db: Session = Depends(get_db)):
    approved_count = 0
    for vid in payload.video_ids:
        video = db.query(Video).filter(Video.video_id == vid).first()
        if video:
            video.approved = True
            video.status = "APPROVED"
            approved_count += 1
    db.commit()
    return {"success": True, "approved_count": approved_count}


@router.post("/api/videos/attach-local/{video_id}")
async def api_attach_local_video(
    video_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    from fastapi import UploadFile, File
    from app.services.downloader_service import DownloaderService
    downloader = DownloaderService()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "read"):
            data = await uploaded_file.read()
            filename = uploaded_file.filename or "video.mp4"
            result = downloader.attach_local_video(db, video_id, data, original_filename=filename)
            return JSONResponse(content=result, status_code=200 if result["success"] else 400)
        source_path = form.get("source_path")
        if source_path:
            result = downloader.attach_local_video(db, video_id, str(source_path).strip())
            return JSONResponse(content=result, status_code=200 if result["success"] else 400)

    try:
        payload = await request.json()
        source_path = payload.get("source_path")
        if source_path:
            result = downloader.attach_local_video(db, video_id, source_path.strip())
            return JSONResponse(content=result, status_code=200 if result["success"] else 400)
    except Exception:
        pass

    return JSONResponse(status_code=400, content={"success": False, "error": "Vui lòng đính kèm file hoặc đường dẫn file hợp lệ."})


@router.post("/api/videos/create-package/{video_id}")
def api_create_package(video_id: str, db: Session = Depends(get_db)):
    from app.services.package_service import CapCutPackageService
    pkg_service = CapCutPackageService()
    result = pkg_service.create_package(db, video_id)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)


@router.get("/api/videos/open-folder/{video_id}")
def api_open_package_folder(video_id: str):
    from app.services.package_service import CapCutPackageService
    pkg_service = CapCutPackageService()
    result = pkg_service.open_package_folder(video_id)
    return JSONResponse(content=result)


@router.post("/api/videos/download/{video_id}")
def api_download_video(video_id: str, db: Session = Depends(get_db)):
    from app.services.downloader_service import DownloaderService
    downloader = DownloaderService()
    result = downloader.download_and_attach(db, video_id=video_id)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)


# ==============================================================================
# Douyin Browser-Assisted Search API Endpoints
# ==============================================================================
class DouyinBrowserSearchRequest(BaseModel):
    keyword: str
    limit: Optional[int] = 10
    provider: Optional[str] = "serpapi"
    force_refresh: Optional[bool] = False


class DouyinBrowserImportItem(BaseModel):
    canonical_url: str
    video_id: Optional[str] = None
    title: Optional[str] = None
    creator: Optional[str] = None
    views: Optional[str] = None
    thumbnail: Optional[str] = None
    source: Optional[str] = None


class DouyinBrowserImportRequest(BaseModel):
    videos: List[DouyinBrowserImportItem]
    product_id: Optional[str] = None


@router.get("/api/douyin-search/providers")
def api_douyin_search_providers():
    from app.services.douyin_search import douyin_search_manager
    from app.config import get_serpapi_api_key
    serpapi_configured = bool(get_serpapi_api_key())
    providers = douyin_search_manager.list_providers()
    for p in providers:
        if p["id"] == "serpapi":
            p["configured"] = serpapi_configured
        else:
            p["configured"] = True
    return JSONResponse(content={"providers": providers})


@router.post("/api/douyin-browser/open")
def api_douyin_browser_open():
    from app.services.douyin_browser_service import douyin_browser_service
    result = douyin_browser_service.open_browser()
    status_code = 200 if result.get("success") else 500
    return JSONResponse(content=result, status_code=status_code)


@router.get("/api/douyin-browser/status")
def api_douyin_browser_status():
    from app.services.douyin_browser_service import douyin_browser_service
    status = douyin_browser_service.get_status()
    return JSONResponse(content=status)


@router.post("/api/douyin-browser/search")
def api_douyin_browser_search(payload: DouyinBrowserSearchRequest, db: Session = Depends(get_db)):
    from app.services.douyin_search import douyin_search_manager
    selected_provider = (payload.provider or "serpapi").lower().strip()
    result = douyin_search_manager.search(
        keyword=payload.keyword,
        limit=payload.limit or 10,
        provider_id=selected_provider,
        db=db,
        force_refresh=bool(payload.force_refresh)
    )
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)



@router.post("/api/douyin-browser/close")
def api_douyin_browser_close():
    from app.services.douyin_browser_service import douyin_browser_service
    result = douyin_browser_service.close_browser()
    return JSONResponse(content=result)


@router.post("/api/douyin-browser/import-selected")
def api_douyin_browser_import_selected(payload: DouyinBrowserImportRequest, db: Session = Depends(get_db)):
    imported_ids: List[str] = []
    duplicate_ids: List[str] = []
    errors: List[str] = []

    for item in payload.videos:
        res = douyin_service.add_video(
            db=db,
            douyin_url=item.canonical_url,
            product_id=payload.product_id if payload.product_id and payload.product_id.strip() else None,
            views=item.views,
            thumbnail=item.thumbnail,
            notes=item.title
        )
        if res.get("success"):
            imported_ids.append(res["video_id"])
        elif res.get("duplicate"):
            duplicate_ids.append(res.get("video_id") or item.canonical_url)
        else:
            errors.append(res.get("error") or "Unknown error")

    return JSONResponse(content={
        "success": len(imported_ids) > 0 or len(duplicate_ids) > 0,
        "imported_count": len(imported_ids),
        "imported_ids": imported_ids,
        "duplicate_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "errors": errors,
        "message": f"Đã nhập {len(imported_ids)} video thành công ({len(duplicate_ids)} video đã tồn tại)."
    })

