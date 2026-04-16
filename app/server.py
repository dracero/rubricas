"""
Simplified FastAPI Server for RubricAI (Skills-based Architecture).

Replaces the multi-process orchestrator + A2A agents with a single server
that uses ADK Runner to process messages through the root agent.
"""

import asyncio
import json
from contextlib import asynccontextmanager
import logging
import os
import sys
import uuid
import shutil
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from dotenv import load_dotenv
load_dotenv()

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.config import setup_langsmith
from app.main_agent import create_root_agent
from app.qdrant_service import TOOL_REGISTRY, _get_qdrant_service
from app.skill_loader import load_skills
from app.models import (
    BatchUploadResponse, FileInfo, RejectedFile,
    BatchStatusResponse, DocumentStatus, DocumentReference, BatchSummary,       
    RubricSummary, RubricDetail, RubricListResponse,
)
from app.rubric_repository import _get_rubric_repository_service
from app.batch_manager import get_batch_manager, ExtractionStatus
from app.ontology_extractor import OntologyExtractor

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.docx_converter import md_to_docx, extract_text_from_docx, detect_rubric_in_response
from app.auth.router import router as auth_router
from app.auth.middleware import auth_middleware
from app.auth.service import get_current_user


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

import tempfile
_tmp = Path(tempfile.gettempdir())
UPLOAD_DIR = _tmp / "rubricas_uploads"
OUTPUT_DIR = _tmp / "rubricas_output"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Skills directory
SKILLS_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of file_id -> original filename for batch uploads
_file_id_to_filename: Dict[str, str] = {}

# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Lifespan context manager: initialize agent on startup."""
    global runner, session_service

    logger.info("=" * 60)
    logger.info("🚀 RubricAI Server v2.0 — Skills Architecture")
    logger.info("=" * 60)

    # Setup LangSmith tracing
    setup_langsmith()

    # Create the root agent with skills
    root_agent = create_root_agent()

    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    logger.info("✅ Server ready — agent and runner initialized")
    yield
    logger.info("🛑 Shutting down RubricAI Server")


app = FastAPI(
    title="RubricAI Server",
    description="Skills-based rubric generation and evaluation system",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "change-me-in-production"),
)
app.middleware("http")(auth_middleware)
app.include_router(auth_router)

# Global state
runner: Runner = None
session_service: InMemorySessionService = None
APP_NAME = "rubricai"
USER_ID = "default_user"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    source: str
    target: str
    type: str
    content: str
    metadata: dict = {}


class GenerateRequest(BaseModel):
    prompt: str
    level: str = "avanzado"
    document_id: str = ""
    document_ids: List[str] = []
    base_rubric_id: Optional[str] = None
    skip_search: bool = False


class EvaluateRequest(BaseModel):
    rubric_id: str
    doc_id: str


# ============================================================================
# PDF Text Extraction
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")






# ============================================================================
# ADK Runner Helper
# ============================================================================

async def run_agent(message: str, session_id: str = None) -> str:
    """Send a message to the root agent via ADK Runner and return the response.

    Args:
        message: The user message to process.
        session_id: Optional session ID for conversation continuity.

    Returns:
        The agent's response text.
    """
    global runner, session_service

    sid = session_id or str(uuid.uuid4())

    # Ensure session exists
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=sid,
    )
    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=sid,
        )

    # Build the user message
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)]
    )

    # Retry configuration for transient errors (503, 429)
    max_retries = 4
    retry_delay = 2.0  # seconds

    for attempt in range(max_retries):
        try:
            # Run the agent and collect response
            response_parts = []
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=sid,
                new_message=user_content,
            ):
                # Debug logging for all events
                author = getattr(event, 'author', '?')
                has_content = event.content is not None
                actions = getattr(event, 'actions', None)
                is_final = getattr(event, 'is_final_response', None)
                
                logger.info(
                    f"📬 Event: author={author}, "
                    f"has_content={has_content}, "
                    f"is_final={is_final}, "
                    f"actions={actions}"
                )
                
                if event.content and event.content.parts:
                    for i, part in enumerate(event.content.parts):
                        has_text = bool(part.text)
                        has_fc = hasattr(part, 'function_call') and part.function_call is not None
                        has_fr = hasattr(part, 'function_response') and part.function_response is not None
                        logger.info(
                            f"  📝 Part[{i}]: text={has_text}, "
                            f"function_call={has_fc}, "
                            f"function_response={has_fr}"
                        )
                        if has_fc:
                            fc = part.function_call
                            logger.info(f"    🔧 Function call: {getattr(fc, 'name', '?')}")
                        if part.text:
                            response_parts.append(part.text)

            if not response_parts:
                logger.warning("⚠️ No text parts collected from any event!")
                return "Sin respuesta del agente."
            
            return "\n".join(response_parts)

        except Exception as e:
            error_str = str(e)
            is_503 = "503" in error_str and "UNAVAILABLE" in error_str
            is_429 = "429" in error_str and "RESOURCE_EXHAUSTED" in error_str

            if (is_503 or is_429) and attempt < max_retries - 1:
                # Extract retry delay from API response if available
                wait_time = retry_delay
                if is_429:
                    import re
                    delay_match = re.search(r'retryDelay.*?(\d+)', error_str)
                    if delay_match:
                        wait_time = max(float(delay_match.group(1)), retry_delay)
                    else:
                        wait_time = max(30.0, retry_delay)  # Default 30s for rate limits

                error_type = "429 RESOURCE_EXHAUSTED" if is_429 else "503 UNAVAILABLE"
                logger.warning(
                    f"⚠️ {error_type}. "
                    f"Reintentando en {wait_time:.0f}s... "
                    f"(Intento {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                retry_delay *= 2  # Exponential backoff for next attempt
                continue
            
            # Re-raise if not retryable or no more retries
            logger.error(f"❌ Error en el agente tras {attempt + 1} intentos: {e}")
            raise e



# ============================================================================
# Startup / Shutdown
# ============================================================================

# Startup is now handled by the lifespan context manager above.


async def _rebuild_agent():
    """Rebuild the root agent and runner with current skills (after upload/delete)."""
    global runner, session_service
    logger.info("🔄 Rebuilding agent with updated skills...")
    root_agent = create_root_agent(str(SKILLS_DIR))
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    logger.info("✅ Agent rebuilt successfully")


# ============================================================================
# API Endpoints - Basic
# ============================================================================

@app.get("/")
async def root():
    """Health check and system info."""
    return {
        "system": "RubricAI Server",
        "version": "2.0.0",
        "architecture": "skills-based",
        "status": "running",
    }


# ============================================================================
# API Endpoints - Chat
# ============================================================================

# Module-level variable to persist chat session across messages
_current_chat_session_id: Optional[str] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint — routes through the root agent."""
    global _current_chat_session_id

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    logger.info(f"📩 Chat received: {user_message[:80]}...")

    # Persist session across chat messages for conversation continuity
    if _current_chat_session_id is None:
        _current_chat_session_id = str(uuid.uuid4())

    try:
        response_text = await run_agent(user_message, session_id=_current_chat_session_id)
    except Exception as e:
        logger.error(f"❌ Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"✅ Agent response: {response_text[:100]}...")

    # Detect action requests for frontend components
    response_type = "text"
    metadata = {"architecture": "skills"}

    # Detect rubric in response and generate DOCX if found
    if detect_rubric_in_response(response_text):
        try:
            docx_filename = f"rubrica_chat_{str(uuid.uuid4())[:8]}.docx"
            md_to_docx(response_text, str(OUTPUT_DIR / docx_filename))
            metadata["download_url"] = f"/api/download/{docx_filename}"
            metadata["has_rubric"] = True
            logger.info(f"📄 Rubric detected in chat response, DOCX generated: {docx_filename}")
        except Exception as e:
            logger.error(f"❌ Error generating DOCX from chat response: {e}")

    # explicitly check for UI tags
    if "[UI:RubricGenerator]" in response_text:
        response_type = "action_request"
        metadata["component"] = "RubricGenerator"
        metadata["routed_to"] = "generator"
        response_text = response_text.replace("[UI:RubricGenerator]", "").strip()
    elif "[UI:RubricEvaluator]" in response_text:
        response_type = "action_request"
        metadata["component"] = "RubricEvaluator"
        metadata["routed_to"] = "evaluator"
        response_text = response_text.replace("[UI:RubricEvaluator]", "").strip()

    # Fallback to heuristic keywords if tag is missing
    if response_type == "text":
        if _needs_generator_action(user_message, response_text):
            response_type = "action_request"
            metadata["component"] = "RubricGenerator"
            metadata["routed_to"] = "generator"
        elif _needs_evaluator_action(user_message, response_text):
            response_type = "action_request"
            metadata["component"] = "RubricEvaluator"
            metadata["routed_to"] = "evaluator"

    return ChatResponse(
        source="orchestrator",
        target="user",
        type=response_type,
        content=response_text,
        metadata=metadata,
    )


def _needs_generator_action(user_msg: str, response: str) -> bool:
    """Detect if the chat response should trigger the generator UI."""
    keywords = ["generar rúbrica", "crear rúbrica", "generar rubrica", "crear rubrica",
                 "genera una rúbrica", "necesito una rúbrica", "generar una rúbrica",
                 "generar una rubrica", "quiero generar", "quiero crear una rúbrica",
                 "generame una rúbrica", "generame una rubrica", "haceme una rúbrica",
                 "haceme una rubrica", "armar una rúbrica", "armar una rubrica",
                 "generar rúbricas", "generar rubricas"]
    msg_lower = user_msg.lower()
    return any(k in msg_lower for k in keywords) or "ACTION:GENERATOR" in response


def _needs_evaluator_action(user_msg: str, response: str) -> bool:
    """Detect if the chat response should trigger the evaluator UI."""
    keywords = ["evaluar documento", "evaluar un documento",
                 "evaluar cumplimiento"]
    msg_lower = user_msg.lower()
    return any(k in msg_lower for k in keywords) or "ACTION:EVALUATOR" in response


# ============================================================================
# API Endpoints - File Upload
# ============================================================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF file for rubric generation."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📤 Uploaded: {file.filename} → {file_id}")
    return {"id": file_id, "filename": file.filename}


# ============================================================================
# API Endpoints - Batch Upload & Status
# ============================================================================

@app.post("/api/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: List[UploadFile] = File(...),
    batch_id: Optional[str] = None,
    clear: bool = True,
):
    """Accept multiple PDF files, store them, and trigger async ontology extraction."""
    accepted: List[FileInfo] = []
    rejected: List[RejectedFile] = []

    # Partition files into accepted (PDF) and rejected (non-PDF)
    for file in files:
        if file.filename and file.filename.lower().endswith(".pdf"):
            file_id = str(uuid.uuid4())
            file_path = UPLOAD_DIR / f"{file_id}.pdf"
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            _file_id_to_filename[file_id] = file.filename
            accepted.append(FileInfo(id=file_id, filename=file.filename))
            logger.info(f"📤 Batch accepted: {file.filename} → {file_id}")
        else:
            rejected.append(
                RejectedFile(
                    filename=file.filename or "unknown",
                    reason="Solo se aceptan archivos PDF",
                )
            )

    bm = get_batch_manager()

    if batch_id and bm.get_batch_status(batch_id):
        # Append to existing batch — do NOT clear Qdrant
        bm.add_documents_to_batch(
            batch_id,
            [f.id for f in accepted],
            [f.filename for f in accepted],
        )
    else:
        # New batch — create and optionally clear Qdrant
        file_ids = [f.id for f in accepted]
        filenames = [f.filename for f in accepted]
        batch_id = bm.create_batch(file_ids, filenames)
        if clear:
            _get_qdrant_service().clear_collection()

    # Spawn async extraction tasks for each accepted document
    for info in accepted:
        file_path = str(UPLOAD_DIR / f"{info.id}.pdf")
        asyncio.create_task(
            _process_document(batch_id, info.id, info.filename, file_path)
        )

    return BatchUploadResponse(
        batch_id=batch_id,
        accepted=accepted,
        rejected=rejected,
    )


@app.get("/api/upload/status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """Return extraction status for each document in the batch."""
    bm = get_batch_manager()
    batch = bm.get_batch_status(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Lote no encontrado")

    documents: List[DocumentStatus] = []
    summary_counts: Dict[str, int] = {
        "completado": 0,
        "en_proceso": 0,
        "error": 0,
        "pendiente": 0,
    }

    for doc in batch.documents.values():
        references = [
            DocumentReference(type=r.type, text=r.text, url=r.url)
            if isinstance(r, DocumentReference)
            else DocumentReference(**r) if isinstance(r, dict) else r
            for r in doc.references
        ]
        documents.append(
            DocumentStatus(
                id=doc.id,
                filename=doc.filename,
                status=doc.status.value,
                entities_count=doc.entities_count,
                relations_count=doc.relations_count,
                error_message=doc.error_message,
                references=references,
            )
        )
        status_key = doc.status.value
        if status_key in summary_counts:
            summary_counts[status_key] += 1

    return BatchStatusResponse(
        batch_id=batch_id,
        documents=documents,
        summary=BatchSummary(
            total=len(documents),
            completado=summary_counts["completado"],
            en_proceso=summary_counts["en_proceso"],
            error=summary_counts["error"],
            pendiente=summary_counts["pendiente"],
        ),
    )


async def _process_document(
    batch_id: str, doc_id: str, filename: str, file_path: str
) -> None:
    """Async extraction task for a single document within a batch.

    Updates BatchManager status throughout the lifecycle and never raises —
    any exception is caught and recorded as an ERROR status.
    """
    bm = get_batch_manager()
    try:
        bm.update_document_status(batch_id, doc_id, ExtractionStatus.EN_PROCESO)

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        if not text or not text.strip():
            bm.update_document_status(
                batch_id,
                doc_id,
                ExtractionStatus.ERROR,
                error_message="Texto vacío o ilegible",
            )
            return

        # Run ontology extraction
        extractor = OntologyExtractor()
        result = await extractor.extract(text, doc_id, filename)

        if not result.success:
            bm.update_document_status(
                batch_id,
                doc_id,
                ExtractionStatus.ERROR,
                error_message=result.error_message or "Error en extracción",
            )
            return

        # Save ontology to Qdrant (additive within the batch)
        _get_qdrant_service().save_ontology_additive(
            result.ontologia, doc_id, filename
        )

        bm.update_document_status(
            batch_id,
            doc_id,
            ExtractionStatus.COMPLETADO,
            entities_count=result.entities_count,
            relations_count=result.relations_count,
            references=result.references,
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error processing document %s (%s): %s",
            filename,
            doc_id,
            exc,
        )
        bm.update_document_status(
            batch_id,
            doc_id,
            ExtractionStatus.ERROR,
            error_message=str(exc),
        )


# ============================================================================
# Multi-document concatenation helper
# ============================================================================

def _concatenate_documents(doc_ids: List[str], max_chars: int = 30000) -> str:
    """Concatenate text from multiple documents with headers and proportional truncation.

    For each document ID, looks up the PDF in UPLOAD_DIR and the original
    filename from _file_id_to_filename.  Texts are joined with
    ``--- DOCUMENTO: {filename} ---`` headers.

    If the combined text exceeds *max_chars*, each document's allocation is
    proportional to its original length (±1 char rounding tolerance).
    """
    texts: List[tuple[str, str]] = []  # (filename, extracted_text)
    for doc_id in doc_ids:
        pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Documento no encontrado: {doc_id}",
            )
        filename = _file_id_to_filename.get(doc_id, f"{doc_id}.pdf")
        text = extract_text_from_pdf(str(pdf_path))
        if text.strip():
            texts.append((filename, text))

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="No se pudo extraer texto de ningún documento",
        )

    # Calculate total raw length (text only, without headers)
    total_len = sum(len(t) for _, t in texts)

    # Build header overhead so we can account for it
    headers = [f"--- DOCUMENTO: {fn} ---\n" for fn, _ in texts]
    header_overhead = sum(len(h) for h in headers)

    if total_len + header_overhead <= max_chars:
        # No truncation needed
        parts = []
        for (fn, text), header in zip(texts, headers):
            parts.append(header + text)
        return "\n".join(parts)

    # Proportional truncation: distribute (max_chars - header_overhead) among docs
    available = max(0, max_chars - header_overhead)
    parts = []
    for (fn, text), header in zip(texts, headers):
        proportion = len(text) / total_len if total_len > 0 else 1.0 / len(texts)
        alloc = int(proportion * available)
        parts.append(header + text[:alloc])
    return "\n".join(parts)


# ============================================================================
# API Endpoints - Generator
# ============================================================================

@app.post("/api/generate")
async def generate_rubric(request: GenerateRequest):
    """Generate a rubric from uploaded PDF document(s)."""

    # --- Multi-document path ---
    if request.document_ids:
        document_text = _concatenate_documents(request.document_ids)
        logger.info(
            f"📄 Concatenated {len(request.document_ids)} documents, "
            f"{len(document_text)} chars"
        )
    else:
        # --- Single-document backward-compat path ---
        pdf_path = UPLOAD_DIR / f"{request.document_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Documento no encontrado.")

        document_text = extract_text_from_pdf(str(pdf_path))
        if not document_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No se pudo extraer texto del PDF",
            )
        logger.info(f"📄 Extracted {len(document_text)} chars from PDF")
        document_text = document_text[:30000]

    # --- Semantic search for similar rubrics ---
    if not request.skip_search:
        try:
            similar = _get_rubric_repository_service().search_similar(document_text[:3000])
            if similar:
                similar_rubrics = [
                    RubricSummary(
                        rubric_id=s["rubric_id"],
                        summary=s["summary"],
                        level=s["level"],
                        source_filenames=s["source_filenames"],
                        created_at=s["created_at"],
                        score=s.get("score"),
                    )
                    for s in similar
                ]
                return {
                    "result": "",
                    "download_url": "",
                    "similar_rubrics": [r.model_dump() for r in similar_rubrics],
                }
        except Exception as e:
            logger.warning(f"⚠️ Semantic search failed, proceeding with generation: {e}")

    # --- Base rubric support ---
    base_rubric_section = ""
    if request.base_rubric_id:
        base_rubric = _get_rubric_repository_service().get_rubric(request.base_rubric_id)
        if not base_rubric:
            raise HTTPException(status_code=404, detail="Rúbrica base no encontrada")
        base_rubric_text = base_rubric.get("rubric_text", "")
        base_rubric_section = (
            f"\n\n--- RÚBRICA BASE DE REFERENCIA ---\n{base_rubric_text}\n\n"
            f"Usa esta rúbrica como base y adaptala al nuevo documento."
        )

    # Build the message for a direct generator agent (bypasses orchestrator/skill transfers)
    agent_message = (
        f"Genera una rúbrica de cumplimiento normativo a partir del siguiente documento.\n"
        f"Nivel de exigencia: {request.level}.\n"
        f"Instrucciones adicionales: {request.prompt}\n\n"
        f"--- TEXTO DEL DOCUMENTO NORMATIVO ---\n{document_text}"
        f"{base_rubric_section}"
    )

    try:
        # Use a dedicated generator agent to avoid orchestrator transfer issues
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm

        generator_agent = Agent(
            name="generador_directo",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            instruction=(
                "Eres un ESPECIALISTA EN COMPLIANCE experto en diseño de instrumentos de evaluación normativa.\n\n"
                "Tu tarea es generar una RÚBRICA DE CUMPLIMIENTO detallada a partir del texto de un documento normativo.\n\n"
                "### Estructura obligatoria de la rúbrica\n\n"
                "1. INFORMACIÓN GENERAL (Ámbito de Aplicación, Nivel de Criticidad, Objetivos)\n"
                "2. ÁREAS DE CUMPLIMIENTO (Requisitos Legales, Operativos, Técnicos, etc.)\n"
                "3. MATRIZ DE EVALUACIÓN. ESTRICTAMENTE EN FORMATO DE TABLA MARKDOWN.\n"
                "   Las columnas de la tabla DEBEN ser: Dimensión | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio.\n"
                "   ASEGÚRATE de que cada fila tenga exactamente 4 celdas separadas por |.\n"
                "   ASEGÚRATE de que la línea separadora (-----|-----|...) tenga también 4 secciones.\n"
                "4. RECOMENDACIONES DE MITIGACIÓN O CORRECCIÓN\n\n"
                "### Reglas Críticas\n"
                "- NO uses términos vagos como 'efectivo' o 'adecuado' sin definirlos.\n"
                "- Cada criterio debe tener EVIDENCIAS OBSERVABLES.\n"
                "- Incluye REQUISITOS MÍNIMOS concretos para aprobar.\n"
                "- Usa formato Markdown.\n"
                "- Sé riguroso, objetivo y profesional.\n"
                "- Responde SOLO con la rúbrica, sin conversación adicional."
            ),
        )

        gen_session_service = InMemorySessionService()
        gen_runner = Runner(
            agent=generator_agent,
            app_name="rubricai_generator",
            session_service=gen_session_service,
        )

        gen_sid = str(uuid.uuid4())
        await gen_session_service.create_session(
            app_name="rubricai_generator",
            user_id=USER_ID,
            session_id=gen_sid,
        )

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=agent_message)]
        )

        response_parts = []
        async for event in gen_runner.run_async(
            user_id=USER_ID,
            session_id=gen_sid,
            new_message=user_content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        rubric_text = "\n".join(response_parts) if response_parts else "Sin respuesta del generador."

        # Clean any UI tags from the response
        rubric_text = rubric_text.replace("[UI:RubricGenerator]", "").replace("[UI:RubricEvaluator]", "").strip()

        output_filename_txt = f"rubrica_{request.document_id[:8]}.txt"
        output_path_txt = OUTPUT_DIR / output_filename_txt
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(rubric_text)

        # Generate DOCX version
        output_filename_docx = f"rubrica_{request.document_id[:8]}.docx"
        output_path_docx = OUTPUT_DIR / output_filename_docx
        md_to_docx(rubric_text, str(output_path_docx))

        logger.info(f"✅ Rubric generated and saved to DOCX: {output_filename_docx}")

        # Store generated rubric in repository (fire-and-forget)
        try:
            doc_ids = request.document_ids if request.document_ids else [request.document_id]
            source_filenames = [_file_id_to_filename.get(did, f"{did}.pdf") for did in doc_ids]
            _get_rubric_repository_service().store_rubric(
                rubric_text, request.level, source_filenames, doc_ids
            )
        except Exception as e:
            logger.error(f"⚠️ Failed to store rubric in repository: {e}")

        return {
            "result": rubric_text,
            "download_url": f"/api/download/{output_filename_docx}",
            "similar_rubrics": [],
        }

    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando rúbrica: {str(e)}")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download a generated rubric file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Detect file extension and set appropriate Content-Type
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == ".pdf":
        media_type = "application/pdf"
    else:
        media_type = "text/plain"

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )


# ============================================================================
# API Endpoints - Rubric Repository
# ============================================================================

@app.get("/api/rubrics")
async def list_rubrics(limit: int = 20, offset: int = 0, search: Optional[str] = None):
    """List rubrics from the repository with optional semantic search."""
    try:
        rubrics, total = _get_rubric_repository_service().list_rubrics(limit, offset, search)
        items = [
            RubricSummary(
                rubric_id=r.get("rubric_id", ""),
                summary=r.get("summary", ""),
                level=r.get("level", ""),
                source_filenames=r.get("source_filenames", []),
                created_at=r.get("created_at", ""),
                score=r.get("score"),
            )
            for r in rubrics
        ]
        return RubricListResponse(
            rubrics=items,
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"❌ Error listing rubrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando rúbricas: {str(e)}")


@app.get("/api/rubrics/{rubric_id}")
async def get_rubric(rubric_id: str):
    """Retrieve a full rubric from the repository."""
    rubric = _get_rubric_repository_service().get_rubric(rubric_id)
    if not rubric:
        raise HTTPException(status_code=404, detail="Rúbrica no encontrada")

    # Generate DOCX
    docx_filename = f"rubrica_repo_{rubric_id[:8]}.docx"
    docx_path = OUTPUT_DIR / docx_filename
    try:
        md_to_docx(rubric.get("rubric_text", ""), str(docx_path))
        download_url = f"/api/download/{docx_filename}"
    except Exception as e:
        logger.error(f"⚠️ Error generating DOCX for rubric {rubric_id}: {e}")
        download_url = None

    return RubricDetail(
        rubric_id=rubric.get("rubric_id", rubric_id),
        rubric_text=rubric.get("rubric_text", ""),
        level=rubric.get("level", ""),
        source_filenames=rubric.get("source_filenames", []),
        source_document_ids=rubric.get("source_document_ids", []),
        created_at=rubric.get("created_at", ""),
        download_url=download_url,
    )


@app.delete("/api/rubrics/{rubric_id}")
async def delete_rubric(rubric_id: str):
    """Delete a rubric from the repository."""
    deleted = _get_rubric_repository_service().delete_rubric(rubric_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rúbrica no encontrada")
    return {"message": f"Rúbrica {rubric_id} eliminada exitosamente"}


# ============================================================================
# API Endpoints - Evaluator
# ============================================================================

@app.post("/api/evaluate/upload_rubric")
async def upload_rubric(file: UploadFile = File(...)):
    """Upload a rubric file for evaluation."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    if ext not in [".txt", ".md", ".pdf", ".docx"]:
        ext = ".txt"
    file_path = UPLOAD_DIR / f"rubric_{file_id}{ext}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📤 Rubric uploaded: {file.filename} → rubric_{file_id}")
    return {"id": file_id, "filename": file.filename}


@app.post("/api/evaluate/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    """Upload a document for evaluation."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".docx"]:
        ext = ".pdf"
    file_path = UPLOAD_DIR / f"doc_{file_id}{ext}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📤 Document uploaded: {file.filename} → doc_{file_id}")
    return {"id": file_id, "filename": file.filename}


@app.post("/api/evaluate/run")
async def run_evaluation(request: EvaluateRequest):
    """Run evaluation: compare a document against a rubric."""
    rubric_path = None
    for ext in [".pdf", ".txt", ".md", ".docx"]:
        candidate = UPLOAD_DIR / f"rubric_{request.rubric_id}{ext}"
        if candidate.exists():
            rubric_path = candidate
            break

    if not rubric_path:
        raise HTTPException(status_code=404, detail="Rúbrica no encontrada")

    doc_path = None
    for ext in [".pdf", ".docx"]:
        candidate = UPLOAD_DIR / f"doc_{request.doc_id}{ext}"
        if candidate.exists():
            doc_path = candidate
            break

    if not doc_path:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    if rubric_path.suffix.lower() == ".pdf":
        rubric_text = extract_text_from_pdf(str(rubric_path))
        if not rubric_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto de la rúbrica PDF")
    elif rubric_path.suffix.lower() == ".docx":
        rubric_text = extract_text_from_docx(str(rubric_path))
        if not rubric_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto de la rúbrica DOCX")
    else:
        rubric_text = rubric_path.read_text(encoding="utf-8")

    if doc_path.suffix.lower() == ".docx":
        document_text = extract_text_from_docx(str(doc_path))
    else:
        document_text = extract_text_from_pdf(str(doc_path))
    if not document_text.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del documento PDF")

    logger.info(f"📋 Evaluating: rubric={len(rubric_text)} chars, doc={len(document_text)} chars")

    # Build the evaluation message with inline text
    agent_message = (
        f"Evalúa el siguiente documento usando esta rúbrica de cumplimiento.\n\n"
        f"--- RÚBRICA ---\n{rubric_text[:20000]}\n\n"
        f"--- DOCUMENTO A EVALUAR ---\n{document_text[:30000]}"
    )

    try:
        # Use a dedicated evaluator agent to bypass orchestrator transfer issues.
        # The orchestrator's transfer mechanism was failing silently (only one
        # model call, no text output). Running the evaluator directly is more
        # reliable since we already know which agent to use.
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm

        evaluator_agent = Agent(
            name="evaluador_directo",
            model=LiteLlm(model="openai/gpt-4o-mini"),
            instruction=(
                "Eres un experto en auditoría y cumplimiento normativo. "
                "Tu tarea es evaluar un documento contra una rúbrica de cumplimiento.\n\n"
                "Para cada criterio de la rúbrica, determina:\n"
                "- **Estado:** Cumple / No Cumple / Parcialmente Cumple\n"
                "- **Evidencia:** Cita textual del documento que justifica el estado\n"
                "- **Observaciones:** Explicación de por qué se asignó ese estado\n"
                "- **Recomendación:** Qué debe cambiar para mejorar\n\n"
                "Al final, presenta un informe con:\n"
                "1. Resumen Ejecutivo (puntaje global o porcentaje de cumplimiento)\n"
                "2. Detalle por Criterio. ESTRICTAMENTE EN FORMATO DE TABLA MARKDOWN.\n"
                "   Las columnas de la tabla DEBEN ser: Dimensión | Criterio de Evaluación | Estado | Evidencia (Cita Textual) | Observaciones | Recomendación.\n"
                "   ASEGÚRATE de que cada fila tenga exactamente 6 celdas separadas por |.\n"
                "   ASEGÚRATE de que la línea separadora (-----|-----|...) tenga también 6 secciones.\n"
                "3. Conclusiones y próximos pasos\n\n"
                "Sé riguroso, objetivo y profesional. Basa tus comentarios "
                "exclusivamente en la evidencia del documento."
            ),
        )

        eval_session_service = InMemorySessionService()
        eval_runner = Runner(
            agent=evaluator_agent,
            app_name="rubricai_evaluator",
            session_service=eval_session_service,
        )

        eval_sid = str(uuid.uuid4())
        await eval_session_service.create_session(
            app_name="rubricai_evaluator",
            user_id=USER_ID,
            session_id=eval_sid,
        )

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=agent_message)]
        )

        response_parts = []
        async for event in eval_runner.run_async(
            user_id=USER_ID,
            session_id=eval_sid,
            new_message=user_content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        eval_text = "\n".join(response_parts) if response_parts else "Sin respuesta del evaluador."

        output_filename_txt = f"evaluacion_{request.doc_id[:8]}.txt"
        output_path_txt = OUTPUT_DIR / output_filename_txt
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(eval_text)

        # Generate DOCX version
        output_filename_docx = f"evaluacion_{request.doc_id[:8]}.docx"
        output_path_docx = OUTPUT_DIR / output_filename_docx
        md_to_docx(eval_text, str(output_path_docx))

        logger.info(f"✅ Evaluation completed and saved to DOCX: {output_filename_docx}")

        return {
            "result": eval_text,
            "download_url": f"/api/download/{output_filename_docx}",
        }

    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluando: {str(e)}")


# ============================================================================
# API Endpoints - Skills Management
# ============================================================================


@app.get("/api/skills/tools")
async def list_available_tools():
    """List all available tools that can be used in skill files."""
    import inspect

    tools_info = []
    for tool_name, tool_func in TOOL_REGISTRY.items():
        # Extract docstring
        doc = inspect.getdoc(tool_func) or "Sin descripción"

        # Extract parameters from signature
        sig = inspect.signature(tool_func)
        params = []
        for param_name, param in sig.parameters.items():
            param_info = {"name": param_name}
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
            if param.default != inspect.Parameter.empty:
                param_info["default"] = str(param.default)
            params.append(param_info)

        tools_info.append({
            "name": tool_name,
            "description": doc,
            "parameters": params,
        })

    return {"tools": tools_info, "total": len(tools_info)}


@app.get("/api/skills")
async def list_skills():
    """List all loaded skills."""
    if not SKILLS_DIR.exists():
        return {"skills": [], "total": 0}

    from google.adk.skills import list_skills_in_dir
    skills = []
    try:
        skills_dict = list_skills_in_dir(SKILLS_DIR)
        for skill_name, fm in skills_dict.items():
            extra_tools = []
            if hasattr(fm, "tools"):
                extra_tools = fm.tools
            elif "tools" in fm.metadata:
                extra_tools = fm.metadata["tools"]
                
            extra_sub = []
            if hasattr(fm, "sub_agents"):
                extra_sub = fm.sub_agents
            elif "sub_agents" in fm.metadata:
                extra_sub = fm.metadata["sub_agents"]

            skills.append({
                "filename": skill_name,  # We use skill_name as the ID in the frontend now
                "name": fm.name,
                "description": fm.description,
                "model": fm.metadata.get("model", getattr(fm, "model", "openai/gpt-4o-mini")),
                "tools": extra_tools,
                "sub_agents": extra_sub,
            })
    except Exception as e:
        logger.error(f"Error listing skills: {e}")

    return {"skills": skills, "total": len(skills)}


@app.post("/api/skills/upload")
async def upload_skill(file: UploadFile = File(...)):
    """Upload a new skill .md file and structure it as an ADK Skill directory."""
    if not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .md")

    content = await file.read()
    content_str = content.decode("utf-8")

    # Validate it has YAML frontmatter
    import re
    import yaml
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content_str, re.DOTALL)
    if not fm_match:
        raise HTTPException(
            status_code=400,
            detail="El archivo debe tener frontmatter YAML (empezar con ---)"
        )

    meta = yaml.safe_load(fm_match.group(1)) or {}
    skill_name = meta.get("name")
    
    if not skill_name:
        # Fallback to sanitized filename if name not provided in FM
        skill_name = file.filename[:-3].replace("_", "-").lower()

    # Enforce ADK kebab-case strict rule (a-z, 0-9, hyphens)
    skill_name = re.sub(r'[^a-z0-9\-]', '-', skill_name.lower())
    skill_name = re.sub(r'-+', '-', skill_name).strip('-')

    if not skill_name:
        raise HTTPException(status_code=400, detail="Nombre del skill invalido o vacio")

    # Structure: skills/<name>/SKILL.md
    skill_dir = SKILLS_DIR / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    file_path = skill_dir / "SKILL.md"

    # Save the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content_str)

    logger.info(f"📄 Skill uploaded and structured in: {skill_dir}")

    # Rebuild the agent with the new skill
    try:
        await _rebuild_agent()
    except Exception as e:
        logger.error(f"Error rebuilding agent: {e}")
        # Don't fail the upload, just warn

    return {
        "filename": skill_name,
        "status": "uploaded",
        "message": f"Skill '{skill_name}' cargado y agente actualizado."
    }


@app.delete("/api/skills/{skill_name}")
async def delete_skill(skill_name: str):
    """Delete an entire skill directory."""
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.exists() or not skill_dir.is_dir():
        raise HTTPException(status_code=404, detail="Skill no encontrado")

    shutil.rmtree(skill_dir)
    logger.info(f"🗑️ Skill deleted: {skill_name}")

    # Rebuild the agent without this skill
    try:
        await _rebuild_agent()
    except Exception as e:
        logger.error(f"Error rebuilding agent: {e}")

    return {"filename": skill_name, "status": "deleted"}


@app.get("/api/skills/{skill_name}/download")
async def download_skill(skill_name: str):
    """Download a skill's SKILL.md file."""
    file_path = SKILLS_DIR / skill_name / "SKILL.md"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Skill no encontrado")
        
    return FileResponse(
        path=str(file_path),
        filename=f"{skill_name}.md",
        media_type="text/markdown",
    )


# ============================================================================
# Uploads / Brand static files
# ============================================================================
BRAND_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "uploads" / "brand"
BRAND_DIR.mkdir(parents=True, exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory=str(BRAND_DIR.parent)), name="uploads")


# ============================================================================
# System status & setup (public — no auth required)
# ============================================================================
from app.auth.db import (
    has_any_admin, get_all_settings, upsert_settings, upsert_setting,
    create_local_user, get_setting,
)
from passlib.context import CryptContext
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


@app.get("/api/system/status")
async def system_status():
    """Check if the system needs first-time setup."""
    try:
        admin_exists = await has_any_admin()
        settings = await get_all_settings()
        setup_done = admin_exists and settings.get("INSTITUCION_NOMBRE", "")
        return {"setup_required": not setup_done}
    except Exception:
        return {"setup_required": True}


@app.post("/api/system/setup")
async def system_setup(
    admin_email: str = Form(None),
    admin_name: str = Form(None),
    admin_password: str = Form(None),
    institucion_nombre: str = Form(None),
    default_language: str = Form("es"),
    logo: UploadFile = File(None),
    background: UploadFile = File(None),
):
    """First-time setup wizard endpoint."""
    # Block if already set up
    admin_exists = await has_any_admin()
    if admin_exists:
        existing = await get_all_settings()
        if existing.get("INSTITUCION_NOMBRE"):
            raise HTTPException(status_code=409, detail="El sistema ya fue configurado.")

    if not admin_email or not admin_password or not institucion_nombre:
        raise HTTPException(status_code=422, detail="Email, contraseña e institución son requeridos.")

    # Create admin user (or promote existing user to admin)
    hashed = _pwd_ctx.hash(admin_password)
    try:
        await create_local_user(admin_email, admin_name or admin_email, hashed, role="admin")
    except Exception:
        # User already exists — update password and promote to admin
        from sqlalchemy import update as sa_update
        from app.auth.db import users as users_table, get_session
        async with await get_session() as session:
            async with session.begin():
                await session.execute(
                    sa_update(users_table).where(users_table.c.email == admin_email).values(
                        hashed_password=hashed, role="admin", name=admin_name or admin_email
                    )
                )

    # Save settings to DB
    await upsert_settings({
        "INSTITUCION_NOMBRE": institucion_nombre,
        "DEFAULT_LANGUAGE": default_language,
    })

    # Save uploaded brand files
    for file_obj, filename in [(logo, "logo.png"), (background, "background.png")]:
        if file_obj and file_obj.size:
            dest = BRAND_DIR / filename
            contents = await file_obj.read()
            dest.write_bytes(contents)
            await upsert_setting(f"BRAND_{filename.split('.')[0].upper()}_URL", f"/uploads/brand/{filename}")

    return {"status": "ok", "message": "Setup completado exitosamente."}


# ============================================================================
# Admin config (read / update)
# ============================================================================
# Editable keys that admins can change at runtime
EDITABLE_KEYS = ["INSTITUCION_NOMBRE", "DEFAULT_LANGUAGE", "AUTH_MODE"]

# Env-only keys shown read-only
ENV_KEYS = [
    "DB_TYPE", "VECTOR_MODE", "QDRANT_URL",
    "ORCHESTRATOR_HOST", "ORCHESTRATOR_PORT",
    "LANGSMITH_PROJECT", "LANGSMITH_TRACING", "FRONTEND_URL",
]
HIDDEN_KEYS = [
    "GOOGLE_API_KEY", "QDRANT_API_KEY", "SECRET_KEY",
    "OPENAI_API_KEY", "LANGSMITH_API_KEY",
]


@app.get("/api/config")
async def get_config(current_user: dict = Depends(get_current_user)):
    """Fetch system configuration. Only accessible by admins."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="No tienes permisos de administrador")

    db_settings = await get_all_settings()

    config_data = {}
    # Editable from DB
    for k in EDITABLE_KEYS:
        config_data[k] = db_settings.get(k, os.getenv(k, ""))
    # Read-only from env
    for k in ENV_KEYS:
        config_data[k] = os.getenv(k, "")
    # Sensitive — masked
    for k in HIDDEN_KEYS:
        config_data[k] = "********" if os.getenv(k) else ""

    # Brand URLs
    config_data["BRAND_LOGO_URL"] = db_settings.get("BRAND_LOGO_URL", "")
    config_data["BRAND_BACKGROUND_URL"] = db_settings.get("BRAND_BACKGROUND_URL", "")

    return {"config": config_data, "editable_keys": EDITABLE_KEYS, "user_role": current_user.get("role")}


@app.put("/api/config/settings")
async def update_settings_json(
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    """Update editable settings (JSON). Only admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="No tienes permisos de administrador")

    updates = {k: v for k, v in body.items() if k in EDITABLE_KEYS and isinstance(v, str)}
    if updates:
        await upsert_settings(updates)
    return {"status": "ok", "updated": list(updates.keys())}


@app.post("/api/config/brand")
async def upload_brand(
    logo: UploadFile = File(None),
    background: UploadFile = File(None),
    current_user: dict = Depends(get_current_user),
):
    """Upload brand files (logo, background). Only admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="No tienes permisos de administrador")

    updated = []
    for file_obj, filename in [(logo, "logo.png"), (background, "background.png")]:
        if file_obj and file_obj.size:
            dest = BRAND_DIR / filename
            contents = await file_obj.read()
            dest.write_bytes(contents)
            key = f"BRAND_{filename.split('.')[0].upper()}_URL"
            await upsert_setting(key, f"/uploads/brand/{filename}")
            updated.append(key)

    return {"status": "ok", "updated": updated}


# ============================================================================
# Public brand info (no auth, used by frontend before login)
# ============================================================================
@app.get("/api/brand")
async def get_brand():
    """Return brand URLs and default language for theming before login."""
    settings = await get_all_settings()
    return {
        "institucion_nombre": settings.get("INSTITUCION_NOMBRE", ""),
        "default_language": settings.get("DEFAULT_LANGUAGE", "es"),
        "logo_url": settings.get("BRAND_LOGO_URL", ""),
        "background_url": settings.get("BRAND_BACKGROUND_URL", ""),
    }


def main():
    import uvicorn
    host = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    port = int(os.getenv("ORCHESTRATOR_PORT", "8000"))

    logger.info(f"🚀 Starting RubricAI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
