"""Rubric Evaluator Agent — ADK Multi-Agent Pipeline.

Uses google.adk Runner + InMemorySessionService to orchestrate:
    root_agent (evaluator)
    └── uses `buscar_contexto_para_evaluacion` tool
"""

import asyncio
import json
import logging
import sys
import os
import time
import re

from typing import Any, Dict

# Add project root to path so we can import common modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Config imports
from common.config import traceable

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/*"]


class RubricEvaluatorAgent:
    """Agent that evaluates documents using the ADK multi-agent pipeline."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/*"]

    def __init__(self):
        """Initialize (lazy-loaded on first invoke)."""
        self._initialized = False
        self._runner = None
        self._session_service = None
        self._root_agent = None

    def _ensure_initialized(self):
        """Lazy-initialize the ADK agent, Runner, and SessionService."""
        if self._initialized:
            return

        logger.info("🔧 Initializing ADK Evaluator pipeline...")

        from common.config import setup_langsmith
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from .adk_agents import create_evaluator_agent

        setup_langsmith()  # Initialize tracing

        self._root_agent = create_evaluator_agent()
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._root_agent,
            app_name="rubric_evaluator",
            session_service=self._session_service,
        )

        self._initialized = True
        logger.info("✅ ADK Evaluator pipeline ready")

    async def _run_agent(self, user_message: str, session_id: str) -> str:
        """Run the ADK agent pipeline and collect the final response.

        Args:
            user_message: The full message to send to the root agent.
            session_id: Session ID for state management.

        Returns:
            The final text response from the agent pipeline.
        """
        from google.genai import types

        # Create session if needed
        session = await self._session_service.get_session(
            app_name="rubric_evaluator",
            user_id="evaluator_user",
            session_id=session_id,
        )
        if session is None:
            session = await self._session_service.create_session(
                app_name="rubric_evaluator",
                user_id="evaluator_user",
                session_id=session_id,
            )

        # Build user content
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )

        # Run the agent and collect events
        text_parts = []
        async for event in self._runner.run_async(
            user_id="evaluator_user",
            session_id=session.id,
            new_message=content,
        ):
            # Debug: log event structure
            role = getattr(event.content, 'role', 'unknown') if event.content else 'event'
            logger.info(f"📡 ADK Evaluator Event: role={role}, has_parts={bool(event.content and event.content.parts)}")

            # Collect text from agent responses
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        # Capture from both 'model' (Gemini) and 'assistant' (Groq)
                        if role in ["model", "assistant"]:
                            logger.info(f"✍️ Collected evaluator text ({len(part.text)} chars)")
                            text_parts.append(part.text)

        final_response = "\n".join(text_parts).strip()
        if not final_response:
            logger.warning("Empty response from ADK Evaluator pipeline")
        return final_response

        return final_response

    @traceable(name="RubricEvaluatorAgent.invoke", run_type="chain")
    def invoke(self, query: str, session_id: str = None) -> str:
        """Evaluate a document against a rubric using ADK agents.

        The query can be either:
        - A JSON message with rubric_text + document_text (from orchestrator)
        - A plain text query (fallback)

        Args:
            query: JSON with rubric_text/document_text or plain text
            session_id: Optional session identifier

        Returns:
            Evaluation result as text
        """
        self._ensure_initialized()

        # Try to parse as structured JSON message from orchestrator
        try:
            data = json.loads(query)
            if isinstance(data, dict) and data.get("type") == "evaluate_document":
                # Truncate to stay under TPM
                # 10k doc + 5k rubric ~ 4k tokens. Safe for 8,000 TPM limit.
                doc_text = data.get("document_text", "")[:10000]
                rub_text = data.get("rubric_text", "")[:5000]
                return self._invoke_with_data(
                    document_text=doc_text,
                    rubric_text=rub_text,
                    session_id=session_id or "default",
                )
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: plain text query
        return self._evaluate_text_query(query)

    def _invoke_with_data(self, rubric_text: str, document_text: str, session_id: str) -> str:
        """Run the ADK pipeline with structured data."""
        logger.info(f"⚖️ ADK Evaluator: doc={len(document_text)} chars, rubric={len(rubric_text)} chars")

        # Build user message for the root agent
        user_message = (
            f"Por favor evalúa el siguiente documento usando la rúbrica proporcionada.\n\n"
            f"RÚBRICA DE REFERENCIA:\n"
            f"{rubric_text[:3000]}\n\n"
            f"DOCUMENTO DEL ESTUDIANTE:\n"
            f"{document_text[:5000]}\n\n"
            f"Instrucciones adicionales: Busca contexto normativo en Qdrant si es necesario "
            f"para validar los criterios de la rúbrica."
        )

        max_retries = 4
        base_wait = 15  # seconds

        for attempt in range(max_retries):
            try:
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run,
                            self._run_agent(user_message, session_id)
                        ).result()
                except RuntimeError:
                    result = asyncio.run(
                        self._run_agent(user_message, session_id)
                    )

                if not result:
                    return "⚠️ El agente no generó una respuesta. Intente nuevamente."

                return result

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "rate_limit_exceeded" in error_str or "RateLimitError" in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    wait = base_wait * (attempt + 1)
                    match = re.search(r"try again in (\d+(?:\.\d+)?)s", error_str)
                    if match:
                        wait = float(match.group(1)) + 2

                    logger.warning(f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue

                logger.exception(f"Evaluator ADK pipeline error: {e}")
                return f"Error al evaluar el documento: {str(e)}"

    def _evaluate_text_query(self, query: str) -> str:
        """Handle a plain text evaluation query (fallback)."""
        logger.info(f"📋 Text query evaluation: {query[:100]}...")
        # Return static guidance to trigger the UI via BeeRouter
        return "¡Entendido! Para realizar una evaluación, necesito que subas dos documentos: la rúbrica de referencia y el trabajo del estudiante. Por favor, usa el formulario a continuación."
