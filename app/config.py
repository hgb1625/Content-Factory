import os
import logging
from pathlib import Path
from typing import Optional
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, override=False):
        p = Path(dotenv_path) if dotenv_path else Path(".env")
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if override or k not in os.environ:
                        os.environ[k] = v

logger = logging.getLogger("app.config")

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

DEFAULT_ACTIVE_AI_PROVIDER = "gemini"

# Production default model: gemini-3.8-flash
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
LEGACY_ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
DEFAULT_MWAPI_MODEL = "claude-sonnet-4-6"

SUPPORTED_AI_PROVIDERS = ["gemini", "openai", "anthropic", "groq", "openrouter", "mwapi"]


def mask_api_key(key: Optional[str] = None) -> str:
    """Mask key securely for safe logging/display. Returns empty string if key is empty."""
    if not key:
        return ""
    k = str(key).strip()
    if len(k) > 8:
        return k[:4] + "*" * (len(k) - 8) + k[-4:]
    return "********"


def get_key_hint(key: Optional[str] = None) -> str:
    """
    Return a non-sensitive key hint (last 4 characters only).
    Never exposes sufficient characters to compromise security.
    """
    if not key:
        return ""
    k = str(key).strip()
    if len(k) >= 8:
        return f"••••{k[-4:]}"
    return "••••••••"


def get_active_ai_provider(db=None) -> str:
    """
    Single source of truth for active AI provider.
    Priority:
    1. SQLite settings table ('active_ai_provider' or 'ACTIVE_AI_PROVIDER') if present and valid.
    2. Environment variable ACTIVE_AI_PROVIDER in .env.
    3. Production default: DEFAULT_ACTIVE_AI_PROVIDER ('gemini').
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["active_ai_provider", "ACTIVE_AI_PROVIDER"])).first()
            if rec and rec.value and rec.value.strip():
                val = rec.value.strip().lower()
                if val in SUPPORTED_AI_PROVIDERS:
                    return val
        except Exception as e:
            logger.debug(f"Could not read active provider from db settings: {e}")
    else:
        try:
            from app.database import SessionLocal
            from app.models import Setting
            with SessionLocal() as session:
                rec = session.query(Setting).filter(Setting.key.in_(["active_ai_provider", "ACTIVE_AI_PROVIDER"])).first()
                if rec and rec.value and rec.value.strip():
                    val = rec.value.strip().lower()
                    if val in SUPPORTED_AI_PROVIDERS:
                        return val
        except Exception as e:
            logger.debug(f"Could not read active provider from db session: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_provider = os.getenv("ACTIVE_AI_PROVIDER", "").strip().lower()
    if env_provider in SUPPORTED_AI_PROVIDERS:
        return env_provider

    return DEFAULT_ACTIVE_AI_PROVIDER


def get_gemini_model(db=None) -> str:
    """
    Single source of truth for active Gemini Model.
    Priority:
    1. SQLite settings table ('gemini_model' or 'GEMINI_MODEL') if present and non-empty.
    2. Environment variable GEMINI_MODEL in .env.
    3. Production default: DEFAULT_GEMINI_MODEL ('gemini-3.8-flash').
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["gemini_model", "GEMINI_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read model from db settings: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model

    return DEFAULT_GEMINI_MODEL


def get_gemini_api_key(db=None) -> str:
    """
    Single source of truth for Gemini API Key.
    Priority:
    1. SQLite settings table ('gemini_api_key' or 'GEMINI_API_KEY') if present.
    2. Environment variable GEMINI_API_KEY in .env.
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["gemini_api_key", "GEMINI_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read api_key from db settings: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("GEMINI_API_KEY", "").strip()


# ==============================================================================
# OPENAI BYOK CONFIGURATION
# ==============================================================================

def get_openai_model(db=None) -> str:
    """Single source of truth for OpenAI Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openai_model", "OPENAI_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openai model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("OPENAI_MODEL", "").strip()
    return env_model or DEFAULT_OPENAI_MODEL


def get_openai_api_key(db=None) -> str:
    """Single source of truth for OpenAI API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openai_api_key", "OPENAI_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openai api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("OPENAI_API_KEY", "").strip()


# ==============================================================================
# ANTHROPIC BYOK CONFIGURATION
# ==============================================================================

def get_anthropic_model(db=None) -> str:
    """Single source of truth for Anthropic Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["anthropic_model", "ANTHROPIC_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                val = rec.value.strip()
                if val == LEGACY_ANTHROPIC_DEFAULT_MODEL:
                    return DEFAULT_ANTHROPIC_MODEL
                return val
        except Exception as e:
            logger.debug(f"Could not read anthropic model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("ANTHROPIC_MODEL", "").strip()
    if env_model == LEGACY_ANTHROPIC_DEFAULT_MODEL:
        return DEFAULT_ANTHROPIC_MODEL
    return env_model or DEFAULT_ANTHROPIC_MODEL


def get_anthropic_api_key(db=None) -> str:
    """Single source of truth for Anthropic API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["anthropic_api_key", "ANTHROPIC_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read anthropic api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


# ==============================================================================
# GROQ BYOK CONFIGURATION
# ==============================================================================

def get_groq_model(db=None) -> str:
    """Single source of truth for Groq Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["groq_model", "GROQ_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read groq model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("GROQ_MODEL", "").strip()
    return env_model or DEFAULT_GROQ_MODEL


def get_groq_api_key(db=None) -> str:
    """Single source of truth for Groq API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["groq_api_key", "GROQ_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read groq api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("GROQ_API_KEY", "").strip()


# ==============================================================================
# OPENROUTER GATEWAY CONFIGURATION
# ==============================================================================

def get_openrouter_model(db=None) -> str:
    """Single source of truth for OpenRouter Gateway Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openrouter_model", "OPENROUTER_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openrouter model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("OPENROUTER_MODEL", "").strip()
    return env_model or DEFAULT_OPENROUTER_MODEL


def get_openrouter_api_key(db=None) -> str:
    """Single source of truth for OpenRouter API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openrouter_api_key", "OPENROUTER_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openrouter api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("OPENROUTER_API_KEY", "").strip()


# ==============================================================================
# MWAPI GATEWAY CONFIGURATION
# ==============================================================================

def get_mwapi_model(db=None) -> str:
    """Single source of truth for MWAPI Gateway Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["mwapi_model", "MWAPI_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read mwapi model from db: {e}")
    else:
        try:
            from app.database import SessionLocal
            from app.models import Setting
            with SessionLocal() as session:
                rec = session.query(Setting).filter(Setting.key.in_(["mwapi_model", "MWAPI_MODEL"])).first()
                if rec and rec.value and rec.value.strip():
                    return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read mwapi model from db session: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("MWAPI_MODEL", "").strip()
    return env_model or DEFAULT_MWAPI_MODEL


def get_mwapi_api_key(db=None) -> str:
    """Single source of truth for MWAPI API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["mwapi_api_key", "MWAPI_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                val = rec.value.strip()
                if "*" not in val and "•" not in val:
                    return val
        except Exception as e:
            logger.debug(f"Could not read mwapi api_key from db: {e}")
    else:
        try:
            from app.database import SessionLocal
            from app.models import Setting
            with SessionLocal() as session:
                rec = session.query(Setting).filter(Setting.key.in_(["mwapi_api_key", "MWAPI_API_KEY"])).first()
                if rec and rec.value and rec.value.strip():
                    val = rec.value.strip()
                    if "*" not in val and "•" not in val:
                        return val
        except Exception as e:
            logger.debug(f"Could not read mwapi api_key from db session: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("MWAPI_API_KEY", "").strip()


# ==============================================================================
# PEXELS VIDEO SOURCE CONFIGURATION
# ==============================================================================

def get_pexels_api_key(db=None) -> str:
    """
    Single source of truth for Pexels API Key.
    Resolution order:
    1. SQLite settings table ('pexels_api_key' or 'PEXELS_API_KEY') if present and non-empty.
    2. Environment variable PEXELS_API_KEY in .env only if SQLite has no key.
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["pexels_api_key", "PEXELS_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                val = rec.value.strip()
                if "*" not in val and "•" not in val:
                    return val
        except Exception as e:
            logger.debug(f"Could not read pexels api_key from db: {e}")
    else:
        try:
            from app.database import SessionLocal
            from app.models import Setting
            with SessionLocal() as session:
                rec = session.query(Setting).filter(Setting.key.in_(["pexels_api_key", "PEXELS_API_KEY"])).first()
                if rec and rec.value and rec.value.strip():
                    val = rec.value.strip()
                    if "*" not in val and "•" not in val:
                        return val
        except Exception as e:
            logger.debug(f"Could not read pexels api_key from db session: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("PEXELS_API_KEY", "").strip()


# ==============================================================================
# MULTI-AI ROUTING & FALLBACK CONFIGURATION (Phase 4 & 5 Hardening)
# ==============================================================================

DEFAULT_AI_FALLBACK_ENABLED = False
DEFAULT_AI_FALLBACK_PROVIDERS = ""
DEFAULT_AI_FALLBACK_ON_QUOTA = False
DEFAULT_AI_FALLBACK_BUDGET_SECONDS = 45.0
MIN_CANDIDATE_TIMEOUT_SECONDS = 3.0
MAX_CANDIDATE_TIMEOUT_SECONDS = 20.0


def get_ai_fallback_enabled(db=None) -> bool:
    """
    Check if cross-provider fallback is explicitly enabled by the customer.
    Defaults to False (off).
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["ai_fallback_enabled", "AI_FALLBACK_ENABLED"])).first()
            if rec and rec.value is not None:
                return str(rec.value).strip().lower() in ("true", "1", "yes", "on")
        except Exception as e:
            logger.debug(f"Could not read ai_fallback_enabled from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_val = os.getenv("AI_FALLBACK_ENABLED", "").strip().lower()
    return env_val in ("true", "1", "yes", "on")


def get_ai_fallback_providers(db=None):
    """
    Retrieve deterministic fallback candidate order as a list of provider IDs.
    Returns only valid supported providers, preserving order.
    """
    raw_str = ""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["ai_fallback_providers", "AI_FALLBACK_PROVIDERS"])).first()
            if rec and rec.value:
                raw_str = str(rec.value).strip()
        except Exception as e:
            logger.debug(f"Could not read ai_fallback_providers from db: {e}")

    if not raw_str:
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        raw_str = os.getenv("AI_FALLBACK_PROVIDERS", "").strip()

    if not raw_str:
        return []

    tokens = [t.strip().lower() for t in raw_str.split(",") if t.strip()]
    valid = []
    for t in tokens:
        if t in SUPPORTED_AI_PROVIDERS and t not in valid:
            valid.append(t)
    return valid


def get_ai_fallback_on_quota(db=None) -> bool:
    """
    Check if cross-provider fallback is allowed on 429 Rate Limit / Quota Exhausted errors.
    Conservative Policy: Defaults to False (off) to prevent unexpected paid quota usage.
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["ai_fallback_on_quota", "AI_FALLBACK_ON_QUOTA"])).first()
            if rec and rec.value is not None:
                return str(rec.value).strip().lower() in ("true", "1", "yes", "on")
        except Exception as e:
            logger.debug(f"Could not read ai_fallback_on_quota from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_val = os.getenv("AI_FALLBACK_ON_QUOTA", "").strip().lower()
    return env_val in ("true", "1", "yes", "on")


def get_ai_fallback_budget_seconds(db=None) -> float:
    """
    Retrieve total wall-clock budget in seconds for an end-to-end routed generation workflow.
    Defaults to 45.0 seconds. Safely normalizes invalid, non-numeric, zero, negative,
    or unreasonably small (< MIN_CANDIDATE_TIMEOUT_SECONDS) values to DEFAULT_AI_FALLBACK_BUDGET_SECONDS.
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["ai_fallback_budget_seconds", "AI_FALLBACK_BUDGET_SECONDS"])).first()
            if rec and rec.value is not None:
                try:
                    val = float(str(rec.value).strip())
                    if val >= MIN_CANDIDATE_TIMEOUT_SECONDS:
                        return val
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            logger.debug(f"Could not read ai_fallback_budget_seconds from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_val = os.getenv("AI_FALLBACK_BUDGET_SECONDS", "").strip()
    if env_val:
        try:
            val = float(env_val)
            if val >= MIN_CANDIDATE_TIMEOUT_SECONDS:
                return val
        except (ValueError, TypeError):
            pass

    return DEFAULT_AI_FALLBACK_BUDGET_SECONDS


def normalize_model_name(provider_id: str, model_name: Optional[str]) -> str:
    """
    Locally normalize model setting for a provider.
    - If model_name is empty or whitespace, returns the provider's known default model.
    - If model_name is non-empty, preserves customer's selection without network verification.
    - If provider is anthropic and model_name is the legacy default, transitions to claude-sonnet-4-6.
    """
    cleaned = (model_name or "").strip()
    pid = (provider_id or "gemini").strip().lower()

    if cleaned:
        if pid == "anthropic" and cleaned == LEGACY_ANTHROPIC_DEFAULT_MODEL:
            return DEFAULT_ANTHROPIC_MODEL
        return cleaned

    if pid == "gemini":
        return DEFAULT_GEMINI_MODEL
    elif pid == "openai":
        return DEFAULT_OPENAI_MODEL
    elif pid == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    elif pid == "groq":
        return DEFAULT_GROQ_MODEL
    elif pid == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    elif pid == "mwapi":
        return DEFAULT_MWAPI_MODEL
    return cleaned or "default"


def get_active_ai_model(db=None) -> str:
    """
    Retrieve the configured model name for the currently active AI provider.
    Guarantees local-only normalization without network calls.
    """
    pid = get_active_ai_provider(db)
    if pid == "openai":
        m = get_openai_model(db)
    elif pid == "anthropic":
        m = get_anthropic_model(db)
    elif pid == "groq":
        m = get_groq_model(db)
    elif pid == "openrouter":
        m = get_openrouter_model(db)
    elif pid == "mwapi":
        m = get_mwapi_model(db)
    else:
        m = get_gemini_model(db)
    return normalize_model_name(pid, m)




