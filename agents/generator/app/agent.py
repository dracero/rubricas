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
import time
import re

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Any, Dict, List

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

        # Run the agent and collect events
        text_parts = []
        async for event in self._runner.run_async(
            user_id="generator_user",
            session_id=session.id,
            new_message=content,
        ):
            # Debug: log event structure
            role = getattr(event.content, 'role', 'unknown') if event.content else 'event'
            logger.info(f"📡 ADK Event: role={role}, has_parts={bool(event.content and event.content.parts)}")

            # Collect text from agent responses
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        # Capture from both 'model' (Gemini) and 'assistant' (Groq)
                        if role in ["model", "assistant"]:
                            logger.info(f"✍️ Collected text ({len(part.text)} chars)")
                            text_parts.append(part.text)

        final_response = "\n".join(text_parts).strip()
        if not final_response:
            logger.warning("Empty response from ADK pipeline")
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
                # Use the full document text now, as chunking will happen in _invoke_with_document
                doc_text = data.get("document_text", "")
                return self._invoke_with_document(
                    document_text=doc_text,
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

    def _chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text.
            chunk_size: Maximum size of each chunk.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
            
        return chunks

    def _invoke_with_document(
        self, document_text: str, prompt: str, level: str, session_id: str
    ) -> str:
        """Processes document in chunks and generates a rubric via ADK pipeline.

        Args:
            document_text: Full text extracted from the normative PDF.
            prompt: User prompt describing the rubric to generate.
            level: Education level (inicial, avanzado, posgrado).
            session_id: Session ID for ADK Runner.

        Returns:
            Generated rubric as text.
        """
        chunks = self._chunk_text(document_text)
        num_chunks = len(chunks)
        logger.info(f"📄 Processing document: {len(document_text)} chars, {num_chunks} chunks, level={level}")

        last_result = ""

        for i, chunk in enumerate(chunks):
            is_last = (i == num_chunks - 1)
            chunk_num = i + 1
            
            # Build instruction for this specific chunk
            if is_last:
                instruction = (
                    f"ÚLTIMO FRAGMENTO (Parte {chunk_num}/{num_chunks}) — AHORA GENERA LA RÚBRICA.\n\n"
                    f"NIVEL EDUCATIVO: {level}\n"
                    f"SOLICITUD: {prompt}\n\n"
                    f"FRAGMENTO FINAL:\n{chunk}\n\n"
                    f"INSTRUCCIONES:\n"
                    f"1. Primero, extrae y guarda la ontología de este fragmento usando la herramienta 'guardar_ontologia_en_qdrant'.\n"
                    f"2. Luego, TRANSFIERE AL AGENTE 'rubricador' para que:\n"
                    f"   - Busque en Qdrant TODO el contexto acumulado de los {num_chunks} fragmentos\n"
                    f"   - Genere la rúbrica completa adaptada al nivel {level}\n"
                    f"   - Cumpla con la solicitud: '{prompt}'\n\n"
                    f"CRÍTICO: Después de guardar la ontología, DEBES transferir al agente 'rubricador' para generar la rúbrica."
                )
            else:
                instruction = (
                    f"ESTE ES UN FRAGMENTO PARCIAL (Parte {chunk_num}/{num_chunks}).\n\n"
                    f"FRAGMENTO DEL DOCUMENTO:\n{chunk}\n\n"
                    f"Tu tarea es ÚNICAMENTE extraer la ontología de este fragmento y guardarla en Qdrant "
                    f"usando la herramienta correspondiente. NO intentes generar la rúbrica todavía ni "
                    f"transfieras al rubricador. Simplemente confirma cuando el guardado sea exitoso."
                )

            logger.info(f"🔄 Processing chunk {chunk_num}/{num_chunks} ({len(chunk)} chars)...")

            max_retries = 4
            base_wait = 15

            for attempt in range(max_retries):
                try:
                    # Use the SAME session ID for all chunks to maintain agent configuration
                    # The session will accumulate history, but Qdrant acts as external memory
                    
                    try:
                        asyncio.get_running_loop()
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            result = pool.submit(
                                asyncio.run,
                                self._run_agent(instruction, session_id)
                            ).result()
                    except RuntimeError:
                        result = asyncio.run(
                            self._run_agent(instruction, session_id)
                        )

                    last_result = result
                    
                    # Add a delay between chunks to respect TPM limits (Groq on_demand is ~30k)
                    if not is_last:
                        delay = 15
                        logger.info(f"⏳ Waiting {delay}s before next chunk to avoid RateLimit...")
                        time.sleep(delay)

                    break # Success, go to next chunk

                except Exception as e:
                    # ... (keep error handling)
                    error_str = str(e)
                    is_rate_limit = "rate_limit_exceeded" in error_str or "RateLimitError" in error_str

                    if is_rate_limit and attempt < max_retries - 1:
                        wait = base_wait * (attempt + 1)
                        match = re.search(r"try again in (\d+(?:\.\d+)?)s", error_str)
                        if match:
                            wait = float(match.group(1)) + 2

                        logger.warning(f"⏳ Rate limit hit on chunk {chunk_num} (attempt {attempt + 1}/{max_retries}). Waiting {wait:.0f}s...")
                        time.sleep(wait)
                        continue

                    logger.exception(f"Error on chunk {chunk_num}: {e}")
                    if is_last:
                        return f"Error al procesar el documento (parte {chunk_num}): {str(e)}"
                    break # Skip this chunk and try next? Or fail fast? Let's skip and hope for the best.

        return last_result
