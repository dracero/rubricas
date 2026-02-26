"""Rubric Evaluator Agent — Migrated to BeeAI.

Calls the pipeline in adk_agents.py.
"""

import asyncio
import json
import logging
import sys
import os

# Add project root to path so we can import common modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from common.config import traceable
from .adk_agents import run_evaluator_pipeline

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/*"]


class RubricEvaluatorAgent:
    """Agent that evaluates documents."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/*"]

    def __init__(self):
        """Initialize."""
        pass

    @traceable(name="RubricEvaluatorAgent.invoke", run_type="chain")
    def invoke(self, query: str, session_id: str = None) -> str:
        """Evaluate a document against a rubric."""
        
        # Try to parse as structured JSON message from orchestrator
        try:
            data = json.loads(query)
            if isinstance(data, dict) and data.get("type") == "evaluate_document":
                return self._invoke_with_data(
                    rubric_text=data.get("rubric_text", ""),
                    document_text=data.get("document_text", ""),
                    session_id=session_id or "default",
                )
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: plain text query
        return self._evaluate_text_query(query)

    def _invoke_with_data(self, rubric_text: str, document_text: str, session_id: str) -> str:
        """Runs the evaluator pipeline."""
        logger.info(f"⚖️ Evaluator pipeline: doc={len(document_text)} chars, rubric={len(rubric_text)} chars")

        try:
            # Run the async pipeline
            try:
                loop = asyncio.get_running_loop()
                # Already in an async context (e.g. uvicorn) — run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        run_evaluator_pipeline(rubric_text, document_text)
                    ).result()
            except RuntimeError:
                # No event loop running — safe to use asyncio.run()
                result = asyncio.run(run_evaluator_pipeline(rubric_text, document_text))

            if not result:
                return "⚠️ El agente no generó una respuesta. Intente nuevamente."

            return result

        except Exception as e:
            logger.exception(f"Evaluator pipeline error: {e}")
            return f"Error al evaluar el documento: {str(e)}"

    def _evaluate_text_query(self, query: str) -> str:
        """Handle a plain text evaluation query (fallback)."""
        logger.info(f"📋 Text query evaluation: {query[:100]}...")
        # Return static guidance to trigger the UI via BeeRouter
        return "¡Entendido! Para realizar una evaluación, necesito que subas dos documentos: la rúbrica de referencia y el trabajo del estudiante. Por favor, usa el formulario a continuación."
