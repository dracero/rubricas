"""Rubric Corrector Agent — ADK Multi-Agent Pipeline.

Uses google.adk Runner + InMemorySessionService to orchestrate:
    root_agent (corrector)
    └── uses `buscar_contexto_para_evaluacion` tool
"""

import asyncio
import json
import logging
import sys
import os

from typing import Any, Dict

# Add project root to path so we can import common modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Config imports
from common.config import traceable

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/*"]


class RubricCorrectorAgent:
    """Agent that corrects and coaches writing using the ADK multi-agent pipeline."""

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

        logger.info("🔧 Initializing ADK Corrector pipeline...")

        from common.config import setup_langsmith
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from .adk_agents import create_corrector_agent

        setup_langsmith()  # Initialize tracing

        self._root_agent = create_corrector_agent()
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._root_agent,
            app_name="rubric_corrector",
            session_service=self._session_service,
        )

        self._initialized = True
        logger.info("✅ ADK Corrector pipeline ready")

    async def _run_agent(self, user_message: str, session_id: str) -> str:
        """Run the ADK agent pipeline and collect the final response."""
        from google.genai import types

        # Create session if needed
        session = await self._session_service.get_session(
            app_name="rubric_corrector",
            user_id="corrector_user",
            session_id=session_id,
        )
        if session is None:
            session = await self._session_service.create_session(
                app_name="rubric_corrector",
                user_id="corrector_user",
                session_id=session_id,
            )

        # Build user content
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )

        # Run the agent and collect events
        final_response = ""
        async for event in self._runner.run_async(
            user_id="corrector_user",
            session_id=session.id,
            new_message=content,
        ):
            # Collect text from agent responses
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        # Only keep the final agent response (last non-tool text)
                        if event.content.role == "model":
                            final_response = part.text

        return final_response

    @traceable(name="RubricCorrectorAgent.invoke", run_type="chain")
    def invoke(self, query: str, session_id: str = None) -> str:
        """Handle a text correction query via ADK agents."""
        self._ensure_initialized()

        # Extract text if JSON
        text_content = query
        try:
            data = json.loads(query)
            if isinstance(data, dict):
                text_content = data.get("text", query)
                session_id = data.get("session_id", session_id)
        except (json.JSONDecodeError, TypeError):
            pass

        return self._invoke_with_text(text_content, session_id or "default")

    def _invoke_with_text(self, text: str, session_id: str) -> str:
        """Run the ADK pipeline with the text."""
        logger.info(f"💬 ADK Corrector: input={len(text)} chars")

        try:
            # Run the async ADK pipeline
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._run_agent(text, session_id)
                    ).result()
            except RuntimeError:
                result = asyncio.run(
                    self._run_agent(text, session_id)
                )

            if not result:
                return "⚠️ El agente no generó una respuesta útil. Intente de nuevo."

            return result

        except Exception as e:
            logger.exception(f"Corrector ADK pipeline error: {e}")
            return f"Error al procesar el texto: {str(e)}"
