import os
import json
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv, set_key

from app.database import get_db, BASE_DIR, TEMP_DIR
from app.config import (
    DEFAULT_GEMINI_MODEL, get_gemini_model, get_gemini_api_key,
    DEFAULT_OPENAI_MODEL, get_openai_model, get_openai_api_key,
    DEFAULT_ANTHROPIC_MODEL, LEGACY_ANTHROPIC_DEFAULT_MODEL, get_anthropic_model, get_anthropic_api_key,
    DEFAULT_GROQ_MODEL, get_groq_model, get_groq_api_key,
    DEFAULT_OPENROUTER_MODEL, get_openrouter_model, get_openrouter_api_key,
    DEFAULT_MWAPI_MODEL, get_mwapi_model, get_mwapi_api_key,
    DEFAULT_ACTIVE_AI_PROVIDER, get_active_ai_provider,
    mask_api_key, get_key_hint, SUPPORTED_AI_PROVIDERS,
    get_ai_fallback_enabled, get_ai_fallback_providers, get_ai_fallback_on_quota,
    get_ai_fallback_budget_seconds, normalize_model_name
)
from app.models import Setting
from app.services.usage_tracker import GeminiUsageTracker
from app.services.subtitle_service import SubtitleService
from app.services.tts.vieneu_provider import VieNeuProvider
from app.services.ai.providers.common import sanitize_secrets
from app.services.ai import get_ai_manager

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
router = APIRouter(tags=["settings"])
ENV_FILE = BASE_DIR / ".env"


class SubtitleSettingsPayload(BaseModel):
    font: Optional[str] = "Arial Bold"
    size: Optional[str] = "medium"
    color: Optional[str] = "#FFFFFF"
    outline_color: Optional[str] = "#000000"
    outline_width: Optional[float] = 2.2
    position: Optional[str] = "bottom"
    custom_y: Optional[float] = 0.84
    animation: Optional[str] = "fade"
    voice_name: Optional[str] = "Trúc Ly"


class PreviewVoicePayload(BaseModel):
    voice_name: Optional[str] = "Trúc Ly"
    text: Optional[str] = "Xin chào, đây là giọng đọc thử nghiệm của hệ thống AI Content Factory."


def save_provider_settings(db: Session, provider_id: str, submitted_key: str, submitted_model: str):
    """
    Persist provider key and model into SQLite Setting (runtime source).
    - Blank or empty submitted_key preserves the existing key in SQLite.
    - Non-empty, non-masked key explicitly replaces the stored key.
    - If provider is Gemini, also syncs to .env for backward compatibility.
    """
    pid = provider_id.strip().lower()
    if pid not in SUPPORTED_AI_PROVIDERS:
        return

    key_clean = submitted_key.strip() if submitted_key else ""
    if key_clean and not key_clean.startswith("****") and "*" not in key_clean and not key_clean.startswith("••••"):
        db_key = db.query(Setting).filter(Setting.key.in_([f"{pid}_api_key", f"{pid.upper()}_API_KEY"])).first()
        if not db_key:
            db_key = Setting(key=f"{pid}_api_key", value=key_clean, description=f"{pid.title()} API Key")
            db.add(db_key)
        else:
            db_key.value = key_clean
        if pid == "gemini":
            set_key(str(ENV_FILE), "GEMINI_API_KEY", key_clean)

    model_clean = submitted_model.strip() if submitted_model else ""
    if pid == "anthropic" and model_clean == LEGACY_ANTHROPIC_DEFAULT_MODEL:
        model_clean = DEFAULT_ANTHROPIC_MODEL
    if model_clean:
        db_model = db.query(Setting).filter(Setting.key.in_([f"{pid}_model", f"{pid.upper()}_MODEL"])).first()
        if not db_model:
            db_model = Setting(key=f"{pid}_model", value=model_clean, description=f"{pid.title()} AI Model")
            db.add(db_model)
        else:
            db_model.value = model_clean
        if pid == "gemini":
            set_key(str(ENV_FILE), "GEMINI_MODEL", model_clean)


def test_provider_connection_logic(provider_id: str, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Execute lightweight, zero-generation-token connection test for a provider.
    Ensures secret sanitization on all messages and error descriptions.
    """
    pid = (provider_id or "gemini").strip().lower()
    if pid not in SUPPORTED_AI_PROVIDERS:
        return {
            "success": False,
            "provider": pid,
            "provider_type": "unknown",
            "configured": False,
            "connected": False,
            "configured_model": "",
            "message": f"Nhà cung cấp AI '{pid}' không được hỗ trợ.",
            "error_type": "unsupported_provider"
        }

    manager = get_ai_manager()
    try:
        report = manager.test_connection(provider_id=pid, db=db)
        if not isinstance(report, dict):
            report = {"connected": False, "message": str(report)}

        connected = bool(report.get("connected", False) or report.get("success", False))
        msg = str(report.get("message") or report.get("error") or "")
        return {
            "success": connected,
            "provider": pid,
            "provider_type": report.get("provider_type", "direct"),
            "configured": bool(report.get("configured", False)),
            "connected": connected,
            "configured_model": report.get("configured_model", report.get("model", "")),
            "message": sanitize_secrets(msg),
            "error_type": report.get("error_type")
        }
    except Exception as e:
        return {
            "success": False,
            "provider": pid,
            "provider_type": "unknown",
            "configured": False,
            "connected": False,
            "configured_model": "",
            "message": sanitize_secrets(str(e)),
            "error_type": "exception"
        }


def get_provider_models_logic(provider_id: str, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Retrieve discovered models for a provider via manager (utilizing ModelCatalogCache).
    """
    pid = (provider_id or "gemini").strip().lower()
    if pid not in SUPPORTED_AI_PROVIDERS:
        return {"success": False, "provider": pid, "models": [], "error": f"Nhà cung cấp '{pid}' không được hỗ trợ."}

    manager = get_ai_manager()
    try:
        models = manager.list_models(provider_id=pid, db=db)
        return {
            "success": True,
            "provider": pid,
            "models": models
        }
    except Exception as e:
        return {
            "success": False,
            "provider": pid,
            "models": [],
            "error": sanitize_secrets(str(e))
        }


def get_ai_system_diagnostics(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Local-only diagnostics for Multi-AI system.
    Strictly performs ZERO network/API calls:
    - Inspects configured status of each provider.
    - Reports active provider, model selections, fallback order and quota policy.
    - Inspects local in-memory catalog cache count if already populated, without fetching.
    """
    from app.services.ai.providers.common import global_model_cache
    active_prov = get_active_ai_provider(db)
    fallback_enabled = get_ai_fallback_enabled(db)
    fallback_providers = get_ai_fallback_providers(db)
    fallback_on_quota = get_ai_fallback_on_quota(db)
    fallback_budget = get_ai_fallback_budget_seconds(db)

    provider_configs = [
        ("gemini", "Google Gemini", "direct", False, get_gemini_api_key, get_gemini_model),
        ("openai", "OpenAI", "direct", False, get_openai_api_key, get_openai_model),
        ("anthropic", "Anthropic", "direct", False, get_anthropic_api_key, get_anthropic_model),
        ("groq", "Groq", "direct", False, get_groq_api_key, get_groq_model),
        ("openrouter", "OpenRouter", "gateway", True, get_openrouter_api_key, get_openrouter_model),
    ]

    providers_status = {}
    for pid, name, ptype, is_gtw, key_fn, model_fn in provider_configs:
        raw_key = key_fn(db) if key_fn else ""
        is_cfg = bool(raw_key and raw_key.strip())
        cfg_model = normalize_model_name(pid, model_fn(db) if model_fn else "")
        hint = get_key_hint(raw_key) if is_cfg else None

        cached_models = global_model_cache.get(pid, raw_key) if is_cfg else None
        cached_count = len(cached_models) if cached_models else 0

        providers_status[pid] = {
            "name": name,
            "provider_type": ptype,
            "is_gateway": is_gtw,
            "configured": is_cfg,
            "key_hint": hint,
            "configured_model": cfg_model,
            "cached_models_count": cached_count,
            "is_active": (pid == active_prov),
            "in_fallback_chain": (pid in fallback_providers)
        }

    return {
        "active_provider": active_prov,
        "fallback_enabled": fallback_enabled,
        "fallback_providers": fallback_providers,
        "fallback_on_quota": fallback_on_quota,
        "fallback_budget_seconds": fallback_budget,
        "providers": providers_status
    }


def get_current_settings(db: Session = None):
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    raw_key = get_gemini_api_key(db)
    masked_key = ""
    if raw_key:
        if len(raw_key) > 8:
            masked_key = raw_key[:4] + "*" * (len(raw_key) - 8) + raw_key[-4:]
        else:
            masked_key = "********"

    usage_stats = GeminiUsageTracker.get_usage(db)
    sub_prefs = SubtitleService.get_user_settings(db)
    vieneu_prov = VieNeuProvider()
    available_voices = vieneu_prov.get_available_voices()
    verified_fonts = SubtitleService.get_verified_fonts()

    from app.services.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, check_ffmpeg_available

    active_prov = get_active_ai_provider(db)
    providers_info = {}
    for pid, name, ptype, is_gtw, key_fn, model_fn, def_model in [
        ("gemini", "Google Gemini", "direct", False, get_gemini_api_key, get_gemini_model, DEFAULT_GEMINI_MODEL),
        ("openai", "OpenAI", "direct", False, get_openai_api_key, get_openai_model, DEFAULT_OPENAI_MODEL),
        ("anthropic", "Anthropic Claude", "direct", False, get_anthropic_api_key, get_anthropic_model, DEFAULT_ANTHROPIC_MODEL),
        ("groq", "Groq", "direct", False, get_groq_api_key, get_groq_model, DEFAULT_GROQ_MODEL),
        ("openrouter", "OpenRouter", "gateway", True, get_openrouter_api_key, get_openrouter_model, DEFAULT_OPENROUTER_MODEL),
        ("mwapi", "MWAPI Gateway", "gateway", True, get_mwapi_api_key, get_mwapi_model, DEFAULT_MWAPI_MODEL),
    ]:
        raw_k = key_fn(db)
        providers_info[pid] = {
            "provider_id": pid,
            "name": name,
            "provider_type": ptype,
            "is_gateway": is_gtw,
            "configured": bool(raw_k),
            "key_hint": get_key_hint(raw_k),
            "model": model_fn(db),
            "default_model": def_model
        }

    fallback_enabled = get_ai_fallback_enabled(db)
    fallback_providers = get_ai_fallback_providers(db)
    fallback_on_quota = get_ai_fallback_on_quota(db)

    from app.services.ai.status import get_active_provider_status
    active_ai_status = get_active_provider_status(db)

    return {
        "active_ai_provider": active_prov,
        "active_ai": active_ai_status,
        "ai_fallback_enabled": fallback_enabled,
        "ai_fallback_providers": fallback_providers,
        "ai_fallback_on_quota": fallback_on_quota,
        "ai_providers": {
            "active_provider": active_prov,
            "providers": providers_info
        },
        "gemini_api_key_masked": masked_key,
        "gemini_api_key_set": bool(raw_key),
        "gemini_key_hint": get_key_hint(raw_key),
        "gemini_model": get_gemini_model(db),
        "gemini_usage": usage_stats,
        "products_per_research": os.getenv("DEFAULT_PRODUCTS_COUNT", "10"),
        "keywords_per_product": os.getenv("DEFAULT_KEYWORDS_COUNT", "3"),
        "videos_per_product": os.getenv("DEFAULT_VIDEOS_COUNT", "5"),
        "tts_engine": os.getenv("TTS_ENGINE", "VieNeu-TTS"),
        "default_voice": sub_prefs.get("voice_name") or os.getenv("DEFAULT_VOICE", "Trúc Ly"),
        "auto_generate_voice": os.getenv("AUTO_GENERATE_VOICE", "true").lower() == "true",
        "original_folder": os.getenv("ORIGINAL_FOLDER", "downloads/original"),
        "package_folder": os.getenv("PACKAGE_FOLDER", "downloads/packages"),
        "final_folder": os.getenv("FINAL_FOLDER", "downloads/final"),
        # Phase 12 Auto Editor Settings
        "ffmpeg_path": os.getenv("FFMPEG_PATH", "") or get_ffmpeg_path() or "",
        "ffprobe_path": os.getenv("FFPROBE_PATH", "") or get_ffprobe_path() or "",
        "auto_edit_cover_type": os.getenv("AUTO_EDIT_COVER_TYPE", "blur"),
        "auto_edit_blur_strength": os.getenv("AUTO_EDIT_BLUR_STRENGTH", "10"),
        "auto_edit_source_audio": os.getenv("AUTO_EDIT_SOURCE_AUDIO", "low"),
        "auto_edit_subtitles": os.getenv("AUTO_EDIT_SUBTITLES", "true").lower() == "true",
        "auto_edit_hook": os.getenv("AUTO_EDIT_HOOK", "true").lower() == "true",
        "auto_edit_auto_render": os.getenv("AUTO_EDIT_AUTO_RENDER", "false").lower() == "true",
        "ffmpeg_status": check_ffmpeg_available(),
        # Phase 13 Subtitle & Voice Settings
        "subtitles": sub_prefs,
        "available_voices": available_voices,
        "verified_fonts": verified_fonts
    }


@router.get("/settings", response_class=HTMLResponse)
def get_settings_page(request: Request, db: Session = Depends(get_db)):
    settings_data = get_current_settings(db)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "settings": settings_data,
            "ai_status": settings_data.get("active_ai"),
            "active_page": "settings",
            "message": None
        }
    )


@router.post("/settings", response_class=HTMLResponse)
def save_settings(
    request: Request,
    active_ai_provider: str = Form("gemini"),
    gemini_api_key: str = Form(""),
    gemini_model: str = Form(DEFAULT_GEMINI_MODEL),
    openai_api_key: str = Form(""),
    openai_model: str = Form(DEFAULT_OPENAI_MODEL),
    anthropic_api_key: str = Form(""),
    anthropic_model: str = Form(DEFAULT_ANTHROPIC_MODEL),
    groq_api_key: str = Form(""),
    groq_model: str = Form(DEFAULT_GROQ_MODEL),
    openrouter_api_key: str = Form(""),
    openrouter_model: str = Form(DEFAULT_OPENROUTER_MODEL),
    mwapi_api_key: str = Form(""),
    mwapi_model: str = Form(DEFAULT_MWAPI_MODEL),
    ai_fallback_enabled: str = Form("off"),
    ai_fallback_providers: str = Form(""),
    ai_fallback_on_quota: str = Form("off"),
    products_per_research: str = Form("10"),
    keywords_per_product: str = Form("3"),
    videos_per_product: str = Form("5"),
    tts_engine: str = Form("VieNeu-TTS"),
    default_voice: str = Form("Trúc Ly"),
    auto_generate_voice: str = Form("off"),
    original_folder: str = Form("downloads/original"),
    package_folder: str = Form("downloads/packages"),
    final_folder: str = Form("downloads/final"),
    ffmpeg_path: str = Form(""),
    ffprobe_path: str = Form(""),
    auto_edit_cover_type: str = Form("blur"),
    auto_edit_blur_strength: str = Form("10"),
    auto_edit_source_audio: str = Form("low"),
    auto_edit_subtitles: str = Form("off"),
    auto_edit_hook: str = Form("off"),
    auto_edit_auto_render: str = Form("off"),
    # Subtitle UX Settings
    subtitle_font: str = Form("Arial Bold"),
    subtitle_size: str = Form("medium"),
    subtitle_color: str = Form("#FFFFFF"),
    subtitle_outline_color: str = Form("#000000"),
    subtitle_outline_width: str = Form("2.2"),
    subtitle_position: str = Form("bottom"),
    subtitle_custom_y: str = Form("0.84"),
    subtitle_animation: str = Form("fade"),
    db: Session = Depends(get_db)
):
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    # 1. Active AI Provider: SQLite Setting is the runtime source
    active_clean = active_ai_provider.strip().lower()
    if active_clean in SUPPORTED_AI_PROVIDERS:
        db_act = db.query(Setting).filter(Setting.key.in_(["active_ai_provider", "ACTIVE_AI_PROVIDER"])).first()
        if not db_act:
            db_act = Setting(key="active_ai_provider", value=active_clean, description="Active AI Provider")
            db.add(db_act)
        else:
            db_act.value = active_clean

    # 2. AI Provider Keys & Models (Blank key preserves existing; new key replaces)
    save_provider_settings(db, "gemini", gemini_api_key, gemini_model)
    save_provider_settings(db, "openai", openai_api_key, openai_model)
    save_provider_settings(db, "anthropic", anthropic_api_key, anthropic_model)
    save_provider_settings(db, "groq", groq_api_key, groq_model)
    save_provider_settings(db, "openrouter", openrouter_api_key, openrouter_model)
    save_provider_settings(db, "mwapi", mwapi_api_key, mwapi_model)

    # 3. Smart Routing & Cross-Provider Fallback Settings
    fallback_en_val = "true" if ai_fallback_enabled in ("on", "true", "1") else "false"
    db_fb_en = db.query(Setting).filter(Setting.key.in_(["ai_fallback_enabled", "AI_FALLBACK_ENABLED"])).first()
    if not db_fb_en:
        db_fb_en = Setting(key="ai_fallback_enabled", value=fallback_en_val, description="AI Fallback Enabled")
        db.add(db_fb_en)
    else:
        db_fb_en.value = fallback_en_val

    fb_tokens = [t.strip().lower() for t in ai_fallback_providers.split(",") if t.strip()]
    fb_valid = [t for t in fb_tokens if t in SUPPORTED_AI_PROVIDERS and t != active_clean]
    fb_val_str = ",".join(fb_valid)
    db_fb_prov = db.query(Setting).filter(Setting.key.in_(["ai_fallback_providers", "AI_FALLBACK_PROVIDERS"])).first()
    if not db_fb_prov:
        db_fb_prov = Setting(key="ai_fallback_providers", value=fb_val_str, description="AI Fallback Candidate Order")
        db.add(db_fb_prov)
    else:
        db_fb_prov.value = fb_val_str

    fallback_quota_val = "true" if ai_fallback_on_quota in ("on", "true", "1") else "false"
    db_fb_q = db.query(Setting).filter(Setting.key.in_(["ai_fallback_on_quota", "AI_FALLBACK_ON_QUOTA"])).first()
    if not db_fb_q:
        db_fb_q = Setting(key="ai_fallback_on_quota", value=fallback_quota_val, description="AI Fallback On Quota (429)")
        db.add(db_fb_q)
    else:
        db_fb_q.value = fallback_quota_val

    db.commit()

    set_key(str(ENV_FILE), "DEFAULT_PRODUCTS_COUNT", products_per_research.strip())
    set_key(str(ENV_FILE), "DEFAULT_KEYWORDS_COUNT", keywords_per_product.strip())
    set_key(str(ENV_FILE), "DEFAULT_VIDEOS_COUNT", videos_per_product.strip())
    set_key(str(ENV_FILE), "TTS_ENGINE", tts_engine.strip())
    set_key(str(ENV_FILE), "DEFAULT_VOICE", default_voice.strip())
    set_key(str(ENV_FILE), "AUTO_GENERATE_VOICE", "true" if auto_generate_voice == "on" else "false")
    set_key(str(ENV_FILE), "ORIGINAL_FOLDER", original_folder.strip())
    set_key(str(ENV_FILE), "PACKAGE_FOLDER", package_folder.strip())
    set_key(str(ENV_FILE), "FINAL_FOLDER", final_folder.strip())

    # Phase 12 Settings
    if ffmpeg_path.strip():
        set_key(str(ENV_FILE), "FFMPEG_PATH", ffmpeg_path.strip())
    if ffprobe_path.strip():
        set_key(str(ENV_FILE), "FFPROBE_PATH", ffprobe_path.strip())

    set_key(str(ENV_FILE), "AUTO_EDIT_COVER_TYPE", auto_edit_cover_type.strip())
    set_key(str(ENV_FILE), "AUTO_EDIT_BLUR_STRENGTH", auto_edit_blur_strength.strip())
    set_key(str(ENV_FILE), "AUTO_EDIT_SOURCE_AUDIO", auto_edit_source_audio.strip())
    set_key(str(ENV_FILE), "AUTO_EDIT_SUBTITLES", "true" if auto_edit_subtitles == "on" else "false")
    set_key(str(ENV_FILE), "AUTO_EDIT_HOOK", "true" if auto_edit_hook == "on" else "false")
    set_key(str(ENV_FILE), "AUTO_EDIT_AUTO_RENDER", "true" if auto_edit_auto_render == "on" else "false")

    # Phase 13 Subtitle Settings
    try:
        sub_width_f = float(subtitle_outline_width.strip())
    except ValueError:
        sub_width_f = 2.2
    try:
        sub_pos_y_f = float(subtitle_custom_y.strip())
    except ValueError:
        sub_pos_y_f = 0.84

    SubtitleService.save_user_settings(db, {
        "font": subtitle_font.strip(),
        "size": subtitle_size.strip(),
        "color": subtitle_color.strip(),
        "outline_color": subtitle_outline_color.strip(),
        "outline_width": sub_width_f,
        "position": subtitle_position.strip(),
        "custom_y": sub_pos_y_f,
        "animation": subtitle_animation.strip(),
        "voice_name": default_voice.strip()
    })

    settings_data = get_current_settings(db)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "settings": settings_data,
            "active_page": "settings",
            "message": "Cài đặt hệ thống & phụ đề đã được lưu thành công!"
        }
    )


@router.get("/api/settings/subtitles")
def api_get_subtitles(db: Session = Depends(get_db)):
    """Retrieve saved subtitle configuration, verified fonts, and available VieNeu voices."""
    sub_prefs = SubtitleService.get_user_settings(db)
    fonts = SubtitleService.get_verified_fonts()
    prov = VieNeuProvider()
    voices = prov.get_available_voices()
    return JSONResponse(content={
        "success": True,
        "settings": sub_prefs,
        "fonts": fonts,
        "voices": voices
    })


@router.post("/api/settings/subtitles")
def api_save_subtitles(payload: SubtitleSettingsPayload, db: Session = Depends(get_db)):
    """Persist subtitle and voice settings into DB."""
    ok = SubtitleService.save_user_settings(db, payload.dict())
    if ok:
        return JSONResponse(content={"success": True, "message": "Cài đặt phụ đề đã được lưu thành công."})
    return JSONResponse(content={"success": False, "error": "Không thể lưu cài đặt phụ đề."}, status_code=500)


@router.get("/api/settings/voices")
def api_get_voices():
    """Retrieve all available VieNeu voices dynamically."""
    prov = VieNeuProvider()
    voices = prov.get_available_voices()
    return JSONResponse(content={"success": True, "voices": voices})


@router.post("/api/settings/preview-voice")
def api_preview_voice(payload: PreviewVoicePayload):
    """Synthesize a short sample sentence to preview the selected voice."""
    prov = VieNeuProvider()
    if not prov.is_available():
        return JSONResponse(content={"success": False, "error": "VieNeu-TTS engine chưa khả dụng."}, status_code=400)

    target_dir = TEMP_DIR / "voice_previews"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"preview_{payload.voice_name.replace(' ', '_')}.wav"

    text = (payload.text or "Xin chào, đây là giọng đọc thử nghiệm.").strip()
    res = prov.synthesize(text=text, output_path=str(out_file), voice_name=payload.voice_name)

    if res.get("success"):
        import time
        audio_url = f"/temp/voice_previews/{out_file.name}?t={int(time.time())}"
        return JSONResponse(content={
            "success": True,
            "voice_name": payload.voice_name,
            "audio_url": audio_url,
            "duration": res.get("duration", 0.0)
        })
    else:
        return JSONResponse(content={"success": False, "error": res.get("error", "Lỗi tạo giọng nói nghe thử")}, status_code=500)


@router.post("/api/settings/test-gemini")
def api_test_gemini():
    from app.services.ai import get_ai_manager
    manager = get_ai_manager()
    return manager.test_connection(provider_id="gemini")


@router.post("/api/settings/test-voice")
def api_test_voice():
    from app.services.tts import TTSService
    tts = TTSService()
    return tts.test_voice()


@router.post("/api/settings/test-watcher")
def api_test_watcher():
    from app.database import FINAL_DIR
    return {
        "success": True,
        "watch_dir": str(FINAL_DIR),
        "exists": FINAL_DIR.exists(),
        "status": "WATCHING"
    }


@router.post("/api/settings/test-ffmpeg")
def api_test_ffmpeg():
    from app.services.ffmpeg_utils import check_ffmpeg_available
    return check_ffmpeg_available()


@router.get("/api/gemini/status")
def api_gemini_status(db: Session = Depends(get_db)):
    from app.services.gemini_status import GeminiStatusTracker
    from app.services.usage_tracker import GeminiUsageTracker
    status_data = GeminiStatusTracker.get_status(db=db)
    usage_data = GeminiUsageTracker.get_usage(db=db)
    return {
        **status_data,
        "requests_today": usage_data["requests_today"],
        "successful_requests": usage_data["successful_requests"],
        "quota_errors": usage_data["quota_errors"],
    }


# ==============================================================================
# MULTI-AI PHASE 3 SETTINGS & DIAGNOSTICS ENDPOINTS
# ==============================================================================

@router.post("/api/settings/ai/test-connection")
async def api_test_ai_connection(request: Request, db: Session = Depends(get_db)):
    """
    Test connection for a specified AI provider (Gemini, OpenAI, Anthropic, Groq, OpenRouter).
    Consumes ZERO generation tokens and sanitizes error responses.
    """
    provider_id = "gemini"
    try:
        data = await request.json()
        if isinstance(data, dict):
            provider_id = data.get("provider_id", data.get("provider", "gemini"))
    except Exception:
        pass

    if not provider_id or provider_id == "gemini":
        qp = request.query_params.get("provider_id") or request.query_params.get("provider")
        if qp:
            provider_id = qp

    return JSONResponse(content=test_provider_connection_logic(provider_id=provider_id, db=db))


@router.get("/api/settings/ai/models")
def api_get_ai_models(provider: Optional[str] = None, provider_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Dynamically retrieve available models for a provider with 60-min in-memory caching.
    """
    pid = (provider_id or provider or "gemini").strip().lower()
    return JSONResponse(content=get_provider_models_logic(provider_id=pid, db=db))


@router.get("/api/settings/ai/status")
def api_get_ai_status(db: Session = Depends(get_db)):
    """
    Return non-sensitive status snapshot of all AI providers and current active provider.
    Guarantees zero raw credentials exposed.
    """
    active_prov = get_active_ai_provider(db)
    providers_status = {}
    for pid, name, ptype, is_gtw, key_fn, model_fn in [
        ("gemini", "Google Gemini", "direct", False, get_gemini_api_key, get_gemini_model),
        ("openai", "OpenAI", "direct", False, get_openai_api_key, get_openai_model),
        ("anthropic", "Anthropic Claude", "direct", False, get_anthropic_api_key, get_anthropic_model),
        ("groq", "Groq", "direct", False, get_groq_api_key, get_groq_model),
        ("openrouter", "OpenRouter", "gateway", True, get_openrouter_api_key, get_openrouter_model),
        ("mwapi", "MWAPI Gateway", "gateway", True, get_mwapi_api_key, get_mwapi_model),
    ]:
        raw_k = key_fn(db)
        providers_status[pid] = {
            "provider_id": pid,
            "name": name,
            "provider_type": ptype,
            "is_gateway": is_gtw,
            "configured": bool(raw_k),
            "key_hint": get_key_hint(raw_k),
            "model": model_fn(db)
        }
    return JSONResponse(content={
        "success": True,
        "active_provider": active_prov,
        "providers": providers_status
    })


@router.get("/api/ai/active-status")
@router.get("/api/ai/status")
def api_get_active_ai_status(db: Session = Depends(get_db)):
    """
    Return local snapshot of the current active AI provider.
    Guarantees ZERO external network calls and ZERO token consumption.
    """
    from app.services.ai.status import get_active_provider_status
    return JSONResponse(content=get_active_provider_status(db=db))


@router.get("/api/settings/ai/diagnostics")
def api_get_ai_diagnostics(db: Session = Depends(get_db)):
    """
    Return local-only system diagnostics for the Multi-AI routing system.
    Guarantees ZERO network calls and ZERO token consumption.
    """
    return JSONResponse(content=get_ai_system_diagnostics(db=db))

