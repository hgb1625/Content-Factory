import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product
from app.services.ai import (
    get_ai_manager,
    AIProviderError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
)
from app.services.ai.base import AIInvalidResponseError
from app.services.gemini_service import (
    GeminiService,
    get_next_product_id,
    GeminiQuotaExceededError,
    sanitize_error_message,
)
from app.services.gemini_status import GeminiStatusTracker
from app.services.niche_catalog import get_catalog_dict, get_broad_categories

logger = logging.getLogger("app.routes.research")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["research"])


def _get_research_provider_info(db: Session) -> dict:
    """Load current active provider info for Research UI — local only, zero network calls."""
    from app.services.ai.status import get_active_provider_status
    st = get_active_provider_status(db=db)
    return {
        "provider_id": st["provider_id"],
        "provider_name": st["display_name"],
        "short_name": st["short_name"],
        "model": st["model"],
        "configured": st["configured"],
        "status": st["status"],
        "status_msg": st["status_msg"],
        "status_hint": st["status_hint"],
        "badge_label": st["badge_label"],
        "badge_color": st["badge_color"],
        "gemini_status": st.get("gemini_status")
    }


def _get_gemini_status(db: Session) -> dict:
    """Load current local Gemini status for display — no network calls."""
    return GeminiStatusTracker.get_status(db=db)


def normalize_niche(niche: str) -> str:
    """Deterministic local normalization for niche strings."""
    if not niche:
        return ""
    return re.sub(r"\s+", " ", niche.strip().lower())


def normalize_product_name(name: str) -> str:
    """Deterministic local normalization for product names (lowercase, stripped punctuation, collapsed whitespace)."""
    if not name:
        return ""
    norm = name.strip().lower()
    norm = re.sub(r"[^\w\s]", "", norm)
    return re.sub(r"\s+", " ", norm).strip()


def get_cached_products_for_niche(db: Session, niche: str) -> List[Product]:
    """Retrieve existing products matching normalized niche without calling AI."""
    norm = normalize_niche(niche)
    if not norm:
        return []
    all_prods = db.query(Product).order_by(Product.id.asc()).all()
    return [p for p in all_prods if normalize_niche(p.niche) == norm]


@router.get("/research", response_class=HTMLResponse)
def get_research_page(
    request: Request,
    niche: Optional[str] = None,
    product_count: Optional[str] = "10",
    fresh: bool = False,
    db: Session = Depends(get_db)
):
    try:
        p_count = int(str(product_count).strip()) if product_count is not None else 10
        if p_count < 1 or p_count > 50:
            p_count = 10
    except (ValueError, TypeError):
        p_count = 10

    cached_count = len(get_cached_products_for_niche(db, niche)) if niche else 0
    prov_info = _get_research_provider_info(db)
    return templates.TemplateResponse(
        request=request,
        name="research.html",
        context={
            "active_page": "research",
            "error": None,
            "error_type": None,
            "error_title": None,
            "success": None,
            "niche_val": niche or "",
            "product_count_val": p_count,
            "fresh_val": fresh,
            "cached_count": cached_count,
            "provider_info": prov_info,
            "ai_status": prov_info,
            "gemini_status": prov_info.get("gemini_status"),
            "niche_catalog": get_catalog_dict(),
            "broad_categories": get_broad_categories(),
            "retry_after": None,
            "request_id": None,
        }
    )



@router.post("/research", response_class=HTMLResponse)
def run_research(
    request: Request,
    niche: str = Form(...),
    product_count: Optional[str] = Form(None),
    fresh: bool = Form(False),
    db: Session = Depends(get_db)
):
    niche_clean = niche.strip()
    norm_niche = normalize_niche(niche_clean)

    # 0. Strict Dual Validation: reject empty/non-numeric, decimal, <1, >50
    # Guaranteed 0 AI generation calls on invalid count.
    raw_count = str(product_count).strip() if product_count is not None else ""
    is_valid_count = False
    valid_count = 10

    if not raw_count or "." in raw_count or "," in raw_count:
        is_valid_count = False
    else:
        try:
            valid_count = int(raw_count)
            if 1 <= valid_count <= 50:
                is_valid_count = True
        except (ValueError, TypeError):
            is_valid_count = False

    if not is_valid_count:
        prov_info = _get_research_provider_info(db)
        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": "Số lượng sản phẩm không hợp lệ. Vui lòng nhập số nguyên từ 1 đến 50.",
                "error_type": "VALIDATION_ERROR",
                "error_title": "Số lượng sản phẩm không hợp lệ",
                "niche_val": niche_clean,
                "product_count_val": raw_count or 10,
                "fresh_val": fresh,
                "provider_info": prov_info,
                "ai_status": prov_info,
                "gemini_status": prov_info.get("gemini_status"),
                "niche_catalog": get_catalog_dict(),
                "broad_categories": get_broad_categories(),
            },
            status_code=400
        )

    # 1. Deterministic Niche Cache Check (0 AI calls if cached results satisfy count)
    if not fresh:
        cached_prods = get_cached_products_for_niche(db, niche_clean)
        if len(cached_prods) >= valid_count:
            logger.info(
                f"Research Cache Hit for niche '{niche_clean}' ({len(cached_prods)} available, "
                f"{valid_count} requested). Reusing existing products with 0 LLM calls."
            )
            return RedirectResponse(
                url=f"/products?success=1&count={valid_count}&cached=1&niche={niche_clean}",
                status_code=303
            )

    # 2. Invoke AI Provider Manager with strict Low-Consumption Mode (exactly ONE LLM attempt)
    ai_manager = get_ai_manager()
    try:
        products_data = ai_manager.generate_products(niche=niche_clean, count=valid_count, db=db)
        if not products_data:
            raise ValueError("Không có sản phẩm nào được tạo ra.")

        # Collect unique products within the generated batch using normalized names
        seen_batch_names = set()
        unique_batch_products = []
        for p in products_data:
            p_name = p.get("name_vietnamese", "").strip()
            norm_name = normalize_product_name(p_name)
            if not norm_name or norm_name in seen_batch_names:
                continue
            seen_batch_names.add(norm_name)
            unique_batch_products.append(p)

        # Enforce exact count invariant: fail if unique valid count is less than requested
        if len(unique_batch_products) < valid_count:
            prov_info = _get_research_provider_info(db)
            raise AIInvalidResponseError(
                f"Phản hồi AI chỉ có {len(unique_batch_products)}/{valid_count} sản phẩm duy nhất hợp lệ (phát hiện tên trùng lặp hoặc thiếu sản phẩm). "
                f"Hệ thống dừng lại và không lưu bản ghi nào để bảo đảm dữ liệu.",
                provider=prov_info.get("provider_id", "unknown")
            )

        if len(unique_batch_products) > valid_count:
            unique_batch_products = unique_batch_products[:valid_count]

        # Existing products in DB for this niche
        existing_names = {
            normalize_product_name(p.name_vietnamese)
            for p in db.query(Product).all()
            if normalize_niche(p.niche) == norm_niche
        }

        created_products = []
        for p in unique_batch_products:
            p_name = p.get("name_vietnamese", "").strip()
            norm_name = normalize_product_name(p_name)
            # Do not allow fresh=True to bypass duplicate-product protection
            if norm_name in existing_names:
                continue

            pid = get_next_product_id(db)
            prod = Product(
                product_id=pid,
                niche=niche_clean,
                name_vietnamese=p_name or "Chưa đặt tên",
                name_chinese=p.get("name_chinese", ""),
                douyin_keywords=p.get("douyin_keywords", ""),
                content_angle=p.get("content_angle", ""),
                hook=p.get("hook", ""),
                status="RESEARCHED"
            )
            db.add(prod)
            db.flush()
            created_products.append(prod)
            existing_names.add(norm_name)

        db.commit()
        count_saved = len(created_products) if created_products else len(unique_batch_products)
        return RedirectResponse(
            url=f"/products?success=1&count={count_saved}&niche={niche_clean}",
            status_code=303
        )

    except (AIQuotaExceededError, GeminiQuotaExceededError) as qe:
        db.rollback()
        clean_qe = sanitize_error_message(str(qe))
        prov_info = _get_research_provider_info(db)
        pname = prov_info["provider_name"]
        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": f"Đã chạm giới hạn hạn mức {pname} (HTTP 429). Research đã dừng và không gửi thêm yêu cầu để bảo vệ hạn ngạch tài khoản. {clean_qe}",
                "error_type": "QUOTA_EXCEEDED",
                "error_title": f"{pname} đã hết hạn mức API / quota",
                "niche_val": niche,
                "product_count_val": valid_count,
                "fresh_val": fresh,
                "provider_info": prov_info,
                "ai_status": prov_info,
                "gemini_status": prov_info.get("gemini_status"),
                "niche_catalog": get_catalog_dict(),
                "broad_categories": get_broad_categories(),
            },
            status_code=429
        )
    except Exception as e:
        db.rollback()
        err_str = sanitize_error_message(str(e))
        prov_info = _get_research_provider_info(db)
        pname = prov_info["provider_name"]
        error_type = "ERROR"
        error_title = f"Lỗi {pname}"
        retry_after = getattr(e, "retry_after", None)
        request_id = getattr(e, "request_id", None)
        duration_sec = getattr(e, "duration_seconds", None)

        if isinstance(e, AIServiceUnavailableError) or "503" in err_str or "quá tải" in err_str.lower() or "overload" in err_str.lower() or "temporarily unavailable" in err_str.lower():
            retry_hint = f" (Nhà cung cấp đề xuất thử lại sau khoảng {retry_after} giây)." if retry_after else ""
            user_msg = (
                f"{pname} đang tạm thời quá tải (HTTP 503). "
                f"Research đã dừng và hệ thống không tự thử lại để tránh phát sinh thêm lượt gọi AI. "
                f"Bạn có thể chờ một lúc rồi thử lại{retry_hint} Với yêu cầu lớn như 30–50 sản phẩm, "
                f"bạn cũng có thể giảm số lượng nếu muốn."
            )
            error_type = "BUSY"
            error_title = f"{pname} đang quá tải"

            # Structured, sanitized 503 diagnostic telemetry (zero secret leakage)
            active_p_id = getattr(e, "provider", None) or prov_info.get("provider_id") or prov_info.get("active_provider_id") or "unknown"
            active_m_name = getattr(e, "model", None) or prov_info.get("model") or prov_info.get("configured_model") or "unknown"
            safe_req_id = sanitize_error_message(str(request_id)) if request_id else "none"
            lat_str = f"{duration_sec:.2f}" if duration_sec is not None else "unknown"
            ra_str = str(retry_after) if retry_after is not None else "none"
            logger.warning(
                f"[RESEARCH 503 OVERLOAD] Provider={active_p_id}, Model={active_m_name}, "
                f"Count={valid_count}, Latency={lat_str}s, RequestID={safe_req_id}, "
                f"RetryAfter={ra_str}s, UpstreamMsg={err_str}"
            )
        elif isinstance(e, AIRateLimitError) or "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
            user_msg = f"{pname} đang giới hạn số yêu cầu tạm thời hoặc đã hết quota (HTTP 429). Research đã dừng và không tự thử lại. Vui lòng đợi vài phút hoặc kiểm tra tài khoản."
            error_type = "RATE_LIMITED"
            error_title = f"{pname} đã hết hạn mức API / quota"
        elif isinstance(e, AIAuthenticationError) or "401" in err_str or "unauthorized" in err_str.lower() or "api key" in err_str.lower():
            user_msg = f"Lỗi {pname} API Key (HTTP 401). Kiểm tra lại API Key trong Cài đặt."
            error_type = "AUTH_ERROR"
            error_title = f"Lỗi xác thực {pname}"
        elif isinstance(e, AIPermissionError) or "403" in err_str or "forbidden" in err_str.lower() or "permission" in err_str.lower():
            user_msg = f"{pname} từ chối truy cập (HTTP 403). Kiểm tra lại quyền hạn của API Key hoặc model trong Cài đặt."
            error_type = "AUTH_ERROR"
            error_title = f"{pname} từ chối truy cập"
        elif isinstance(e, AIModelNotFoundError) or "404" in err_str or "model_not_found" in err_str.lower():
            user_msg = f"Không tìm thấy mô hình hoặc mô hình không được hỗ trợ trên {pname}."
            error_type = "ERROR"
            error_title = f"Không tìm thấy mô hình {pname}"
        elif isinstance(e, AITimeoutError) or "timed out" in err_str.lower() or "timeout" in err_str.lower():
            user_msg = f"Yêu cầu tới {pname} bị quá thời gian chờ (Timeout). Research đã dừng và không tự thử lại để tiết kiệm hạn mức."
            error_type = "NETWORK_ERROR"
            error_title = f"{pname} hết thời gian chờ"
        elif isinstance(e, AINetworkError) or "network" in err_str.lower() or "connection" in err_str.lower():
            user_msg = f"Không kết nối được {pname}. Research đã dừng và không tự thử lại. Vui lòng kiểm tra kết nối mạng."
            error_type = "NETWORK_ERROR"
            error_title = f"Không kết nối được {pname}"
        elif isinstance(e, AIInvalidResponseError) or "json" in err_str.lower() or "malformed" in err_str.lower():
            if "finish_reason='length'" in err_str or "giới hạn token" in err_str.lower():
                user_msg = (
                    f"{pname} đã tạo nội dung nhưng bị cắt ngang do chạm giới hạn token (finish_reason='length'). "
                    f"Hệ thống đã dừng lại để bảo toàn dữ liệu. Bạn có thể thử lại hoặc giảm số lượng sản phẩm."
                )
                error_type = "ERROR"
                error_title = f"{pname} chạm giới hạn token"
            elif "sản phẩm được yêu cầu" in err_str.lower():
                user_msg = (
                    f"{pname} trả về số lượng sản phẩm không khớp với yêu cầu: {err_str} "
                    f"Hệ thống đã dừng lại và không tự gọi lại để tránh phát sinh chi phí."
                )
                error_type = "ERROR"
                error_title = f"Số lượng sản phẩm {pname} không khớp"
            else:
                user_msg = f"{pname} trả về dữ liệu không hợp lệ. Hệ thống không tự tạo lại để tránh tốn thêm quota: {err_str}"
                error_type = "ERROR"
                error_title = f"Dữ liệu {pname} không hợp lệ"
        else:
            user_msg = f"Lỗi Research ({pname}): {err_str}"
            error_title = f"Lỗi {pname}"

        resp_status = 503 if error_type == "BUSY" else 400
        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": user_msg,
                "error_type": error_type,
                "error_title": error_title,
                "niche_val": niche,
                "product_count_val": valid_count,
                "fresh_val": fresh,
                "provider_info": prov_info,
                "ai_status": prov_info,
                "gemini_status": prov_info.get("gemini_status"),
                "niche_catalog": get_catalog_dict(),
                "broad_categories": get_broad_categories(),
                "retry_after": retry_after,
                "request_id": request_id,
            },
            status_code=resp_status
        )

