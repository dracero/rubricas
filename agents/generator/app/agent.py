"""Rubric Generator Agent — Migrated to BeeAI.

Calls the pipeline in adk_agents.py.
"""

import asyncio
import json
import logging
import sys
import os

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from common.config import traceable
from .adk_agents import run_generator_pipeline

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/*"]


class RubricGeneratorAgent:
    """Agent that generates academic rubrics."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/*"]

    def __init__(self):
        """Initialize."""
        pass

    @traceable(name="RubricGeneratorAgent.invoke", run_type="chain")
    def invoke(self, query: str, session_id: str = None) -> str:
        """Generate a rubric based on the user query."""
        
        # Try to parse as structured JSON message from orchestrator
        try:
            data = json.loads(query)
            if isinstance(data, dict) and data.get("type") == "generate_rubric":
                return self._invoke_with_document(
                    document_text=data.get("document_text", ""),
                    prompt=data.get("prompt", "Generar rúbrica"),
                    level=data.get("level", "avanzado"),
                    session_id=session_id or "default",
                    llm_config=data.get("llm_config"),
                )
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: plain text query
        return (
            "¡Claro! Para generar una rúbrica personalizada, por favor utiliza "
            "el formulario para subir tu documento normativo (PDF) y selecciona "
            "el nivel educativo. Si prefieres, también puedes describirme el tema "
            "directamente aquí."
        )

    def _invoke_with_document(
        self, document_text: str, prompt: str, level: str, session_id: str,
        llm_config: dict = None,
    ) -> str:
        """Runs the generator pipeline."""
        
        logger.info(f"📝 Generator pipeline: doc={len(document_text)} chars, level={level}")

        try:
            # Run the async pipeline
            try:
                loop = asyncio.get_running_loop()
                # Already in an async context (e.g. uvicorn) — run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        run_generator_pipeline(document_text, prompt, level, llm_config)
                    ).result()
            except RuntimeError:
                # No event loop running — safe to use asyncio.run()
                result = asyncio.run(run_generator_pipeline(document_text, prompt, level, llm_config))

            if not result:
                return "⚠️ El agente no generó una respuesta. Intente nuevamente."

            return result

        except Exception as e:
            logger.exception(f"Generator pipeline error: {e}")
            return f"Error al generar la rúbrica: {str(e)}"

