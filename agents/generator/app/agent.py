"""Rubric Generator Agent — ADK Multi-Agent Pipeline.

Uses google.adk Runner + InMemorySessionService to orchestrate:
    root_agent (orchestrator)
    ├── ontologo_agent  (extracts ontology → saves to Qdrant)
    └── rubricador_agent (RAG from Qdrant → generates rubric)
"""

import asyncio
import json
import logging
import sys
import os

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Any, Dict

from common.config import traceable

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/*"]


class RubricGeneratorAgent:
    """Agent that generates academic rubrics using the ADK multi-agent pipeline."""

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

        logger.info("🔧 Initializing ADK Generator pipeline...")

        from common.config import setup_langsmith
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from .adk_agents import create_generator_agent

        setup_langsmith()  # Initialize tracing

        self._root_agent = create_generator_agent()
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._root_agent,
            app_name="rubric_generator",
            session_service=self._session_service,
        )

        self._initialized = True
        logger.info("✅ ADK Generator pipeline ready")

    async def _run_agent(self, user_message: str, session_id: str) -> str:
        """Run the ADK agent pipeline and collect the final response.

        Args:
            user_message: The full message to send to the root agent.
            session_id: Session ID for state management.

        Returns:
            The final text response from the agent pipeline.
        """
        from google.genai import types
        from common.config import get_current_run_tree

        # Create session if needed
        session = await self._session_service.get_session(
            app_name="rubric_generator",
            user_id="generator_user",
            session_id=session_id,
        )
        if session is None:
            session = await self._session_service.create_session(
                app_name="rubric_generator",
                user_id="generator_user",
                session_id=session_id,
            )

        # Build user content
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )

        # Track agent execution metadata
        agent_steps = []
        total_tokens = 0
        llm_calls = 0

        # Run the agent and collect events
        final_response = ""
        async for event in self._runner.run_async(
            user_id="generator_user",
            session_id=session.id,
            new_message=content,
        ):
            # Track agent steps and LLM interactions
            if event.content:
                llm_calls += 1
                step_info = {
                    "step": llm_calls,
                    "role": event.content.role,
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else None,
                }
                
                # Estimate tokens (rough approximation)
                if event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Rough token estimation: ~4 chars per token
                            estimated_tokens = len(part.text) // 4
                            total_tokens += estimated_tokens
                            step_info["estimated_tokens"] = estimated_tokens
                            
                            # Only keep the final agent response (last non-tool text)
                            if event.content.role == "model":
                                final_response = part.text
                
                agent_steps.append(step_info)

        # Log comprehensive metadata to LangSmith
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra = run_tree.extra or {}
            run_tree.extra.update({
                "agent_type": "ADK Multi-Agent",
                "session_id": session_id,
                "total_llm_calls": llm_calls,
                "estimated_total_tokens": total_tokens,
                "agent_steps": agent_steps,
                "sub_agents": ["ontologo", "rubricador"],
                "message_length": len(user_message),
            })

        return final_response

    @traceable(name="RubricGeneratorAgent.invoke", run_type="chain")
    def invoke(self, query: str, session_id: str = None) -> str:
        """Generate a rubric based on the user query.

        The query can be either:
        - A JSON message with document_text, level, and prompt (full pipeline)
        - A plain text prompt (returns guidance message)

        Args:
            query: User prompt or JSON with document_text/level/prompt
            session_id: Optional session identifier

        Returns:
            Generated rubric as text
        """
        self._ensure_initialized()

        # Try to parse as structured JSON message from orchestrator
        try:
            data = json.loads(query)
            if isinstance(data, dict) and data.get("type") == "generate_rubric":
                return self._invoke_with_document(
                    document_text=data.get("document_text", ""),
                    prompt=data.get("prompt", "Generar rúbrica"),
                    level=data.get("level", "avanzado"),
                    session_id=session_id or "default",
                )
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: plain text query — disabled to prevent premature generation
        return (
            "¡Claro! Para generar una rúbrica personalizada, por favor utiliza "
            "el formulario para subir tu documento normativo (PDF) y selecciona "
            "el nivel educativo. Si prefieres, también puedes describirme el tema "
            "directamente aquí."
        )

    def _invoke_with_document(
        self, document_text: str, prompt: str, level: str, session_id: str
    ) -> str:
        """Full ADK pipeline: sends document + prompt to the ADK multi-agent system.

        The root agent will:
        1. Clear the Qdrant database to start fresh
        2. Delegate to ontólogo → extracts ontology → saves to Qdrant
        3. Delegate to rubricador → searches Qdrant → generates rubric

        Args:
            document_text: Text extracted from the normative PDF
            prompt: User prompt describing the rubric to generate
            level: Education level (inicial, avanzado, posgrado)
            session_id: Session ID for ADK Runner

        Returns:
            Generated rubric as text
        """
        logger.info(f"📝 ADK pipeline: doc={len(document_text)} chars, level={level}")
        
        # Clear Qdrant database before processing new document
        try:
            from .adk_agents import _get_qdrant_service
            qdrant_service = _get_qdrant_service()
            qdrant_service.clear_collection()
        except Exception as e:
            logger.warning(f"⚠️ Could not clear Qdrant collection: {e}")

        # Build user message for the root agent
        user_message = (
            f"Necesito generar una rúbrica de evaluación.\n\n"
            f"NIVEL EDUCATIVO: {level}\n\n"
            f"SOLICITUD: {prompt}\n\n"
            f"DOCUMENTO NORMATIVO:\n"
            f"{document_text[:20000]}\n\n"
            f"Por favor:\n"
            f"1. Primero usa al ontólogo para extraer la ontología del documento "
            f"y guardarla en Qdrant.\n"
            f"2. Luego usa al rubricador para buscar contexto en Qdrant y generar "
            f"la rúbrica adaptada al nivel {level}."
        )

        try:
            # Run the async ADK pipeline
            try:
                loop = asyncio.get_running_loop()
                # Already in an async context (e.g. uvicorn) — run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._run_agent(user_message, session_id)
                    ).result()
            except RuntimeError:
                # No event loop running — safe to use asyncio.run()
                result = asyncio.run(
                    self._run_agent(user_message, session_id)
                )

            if not result:
                return "⚠️ El agente no generó una respuesta. Intente nuevamente."

            return result

        except Exception as e:
            logger.exception(f"Generator ADK pipeline error: {e}")
            return f"Error al generar la rúbrica: {str(e)}"
