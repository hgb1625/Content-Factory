"""
Multi-AI Architecture Package (Phase 1)

Exports core contracts, manager singleton, and provider adapters.
"""
from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
)
from app.services.ai.manager import (
    AIProviderManager,
    get_ai_manager,
    set_ai_manager,
)
from app.services.ai.providers.gemini import GeminiProvider
from app.services.ai.status import get_active_provider_status

__all__ = [
    "AIProvider",
    "AIGenerationOptions",
    "ExecutionMetadata",
    "AIProviderManager",
    "get_ai_manager",
    "set_ai_manager",
    "GeminiProvider",
    "AIProviderError",
    "AIAuthenticationError",
    "AIPermissionError",
    "AIModelNotFoundError",
    "AIQuotaExceededError",
    "AIRateLimitError",
    "AIServiceUnavailableError",
    "AITimeoutError",
    "AINetworkError",
]
