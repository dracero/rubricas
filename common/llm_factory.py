"""
LLM Factory — Provider abstraction and model catalog.

Creates BeeAI ChatModel instances dynamically based on provider/model/api_key.
Falls back to environment variables when api_key is not provided.
"""

import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CATALOG
# ============================================================================

LLM_CATALOG = {
    "groq": {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "models": [
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout 17B"},
            {"id": "meta-llama/llama-4-maverick-17b-128e-instruct", "name": "Llama 4 Maverick 17B"},
            {"id": "meta-llama/llama-3.3-70b-versatile", "name": "Llama 3.3 70B"},
            {"id": "deepseek-r1-distill-llama-70b", "name": "DeepSeek R1 Distill 70B"},
            {"id": "qwen-qwq-32b", "name": "Qwen QwQ 32B"},
        ],
    },
    "google": {
        "name": "Google Gemini",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
        ],
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
            {"id": "gpt-4.1", "name": "GPT-4.1"},
            {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini"},
        ],
    },
}

# Default configuration (used when nothing is specified)
DEFAULT_LLM_CONFIG = {
    "provider": "groq",
    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    "api_key": "",  # Empty = fall back to env var
}


def get_catalog() -> dict:
    """Return the model catalog for the frontend."""
    return LLM_CATALOG


def create_llm(
    provider: str = "groq",
    model_id: str = "",
    api_key: str = "",
) -> Any:
    """Create a BeeAI ChatModel instance based on provider and model.

    Args:
        provider: Provider key from LLM_CATALOG (e.g. "groq", "google", "openai")
        model_id: Model identifier (e.g. "gemini-2.5-flash")
        api_key: API key. If empty, falls back to the provider's env variable.

    Returns:
        A BeeAI ChatModel instance ready to use with ReActAgent.
    """
    if provider not in LLM_CATALOG:
        raise ValueError(f"Proveedor no soportado: {provider}. Opciones: {list(LLM_CATALOG.keys())}")

    catalog_entry = LLM_CATALOG[provider]

    # Resolve API key: user-provided > env var
    resolved_key = api_key or os.environ.get(catalog_entry["env_key"], "")
    if not resolved_key:
        logger.warning(f"⚠️ No API key found for {provider} (env: {catalog_entry['env_key']})")

    # Default model if not specified
    if not model_id:
        model_id = catalog_entry["models"][0]["id"]

    logger.info(f"🤖 Creating LLM: provider={provider}, model={model_id}")

    if provider == "groq":
        from beeai_framework.adapters.groq import GroqChatModel
        return GroqChatModel(model_id=model_id, api_key=resolved_key)

    elif provider == "google":
        from beeai_framework.adapters.gemini import GeminiChatModel
        return GeminiChatModel(model_id=model_id, api_key=resolved_key)

    elif provider == "openai":
        from beeai_framework.adapters.openai import OpenAIChatModel
        return OpenAIChatModel(model_id=model_id, api_key=resolved_key)

    else:
        raise ValueError(f"Proveedor '{provider}' no tiene implementación.")


def resolve_llm_config(llm_config: Optional[dict] = None) -> dict:
    """Merge user-provided config with defaults.

    Returns a complete config dict with provider, model_id, and api_key.
    """
    if not llm_config:
        return DEFAULT_LLM_CONFIG.copy()

    return {
        "provider": llm_config.get("provider") or DEFAULT_LLM_CONFIG["provider"],
        "model_id": llm_config.get("model_id") or DEFAULT_LLM_CONFIG["model_id"],
        "api_key": llm_config.get("api_key") or "",
    }
