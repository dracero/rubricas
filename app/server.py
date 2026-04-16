"""
Simplified FastAPI Server for RubricAI (Skills-based Architecture).

Replaces the multi-process orchestrator + A2A agents with a single server
that uses ADK Runner to process messages through the root agent.
"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime
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

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
from app.i18n import get_request_language, get_message, LANGUAGE_NAMES


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

# Rubric repository directory (persistent local storage)
RUBRICS_REPO_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "rubrics_repo"
RUBRICS_REPO_DIR.mkdir(parents=True, exist_ok=True)

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
    logger.info("🚀 AsistIAG Server v2.0 — Skills Architecture")
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
    logger.info("🛑 Shutting down AsistIAG Server")


app = FastAPI(
    title="AsistIAG Server",
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
    rubric_id: str = ""
    rubric_filename: str = ""
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

async def run_agent(message: str, session_id: str = None, lang: str = 'es') -> str:
    """Send a message to the root agent via ADK Runner and return the response.

    Args:
        message: The user message to process.
        session_id: Optional session ID for conversation continuity.
        lang: Language code for the response (default 'es').

    Returns:
        The agent's response text.
    """
    global runner, session_service

    sid = session_id or str(uuid.uuid4())

    # Prepend language directive so the agent responds in the requested language
    language_name = LANGUAGE_NAMES.get(lang, 'español')
    prefixed_message = f"[SYSTEM: Respond in {language_name}. All your output must be in {language_name}.]\n\n{message}"

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
        parts=[types.Part.from_text(text=prefixed_message)]
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
        "system": "AsistIAG Server",
        "version": "2.0.0",
        "architecture": "skills-based",
        "status": "running",
    }


@app.get("/api/config")
async def get_config():
    """Return system configuration for the frontend."""
    return {
        "language": os.getenv("SYSTEM_LANGUAGE", "es"),
        "institution": os.getenv("INSTITUTION_NAME", ""),
    }


# ============================================================================
# API Endpoints - Chat
# ============================================================================

# Module-level variable to persist chat session across messages
_current_chat_session_id: Optional[str] = None


@app.post("/api/chat/reset")
async def reset_chat():
    """Reset the chat session (e.g. when language changes)."""
    global _current_chat_session_id
    _current_chat_session_id = None
    logger.info("🔄 Chat session reset")
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    """Main chat endpoint — routes through the root agent."""
    global _current_chat_session_id

    lang = get_request_language(request)
    user_message = chat_request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail=get_message('empty_message', lang))

    logger.info(f"📩 Chat received: {user_message[:80]}...")

    # Persist session across chat messages for conversation continuity
    if _current_chat_session_id is None:
        _current_chat_session_id = str(uuid.uuid4())

    try:
        response_text = await run_agent(user_message, session_id=_current_chat_session_id, lang=lang)
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
    elif "[UI:RubricRepository]" in response_text:
        response_type = "action_request"
        metadata["component"] = "RubricRepository"
        metadata["routed_to"] = "repository"
        response_text = response_text.replace("[UI:RubricRepository]", "").strip()

    # Use LLM to classify intent if no UI tag was found
    if response_type == "text":
        intent = _classify_intent(user_message)
        if intent == "generator":
            response_type = "action_request"
            metadata["component"] = "RubricGenerator"
            metadata["routed_to"] = "generator"
        elif intent == "evaluator":
            response_type = "action_request"
            metadata["component"] = "RubricEvaluator"
            metadata["routed_to"] = "evaluator"
        elif intent == "repository":
            response_type = "action_request"
            metadata["component"] = "RubricRepository"
            metadata["routed_to"] = "repository"

    return ChatResponse(
        source="orchestrator",
        target="user",
        type=response_type,
        content=response_text,
        metadata=metadata,
    )


def _classify_intent(user_msg: str) -> str:
    """Use LLM to classify user intent into: generator, evaluator, repository, or chat."""
    try:
        import litellm
        response = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Clasifica el intent del usuario en exactamente UNA de estas categorías:\n"
                    "- generator: quiere CREAR o GENERAR una rúbrica nueva a partir de un documento\n"
                    "- evaluator: quiere EVALUAR un documento contra una rúbrica existente\n"
                    "- repository: quiere VER, BUSCAR, LISTAR, MODIFICAR o GESTIONAR rúbricas guardadas en el repositorio\n"
                    "- chat: conversación general, preguntas, o cualquier otra cosa\n\n"
                    "Responde SOLO con una palabra: generator, evaluator, repository, o chat"
                )},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        intent = response.choices[0].message.content.strip().lower()
        if intent in ("generator", "evaluator", "repository"):
            return intent
    except Exception as e:
        logger.warning(f"⚠️ Intent classification failed: {e}")
    return "chat"


# ============================================================================
# API Endpoints - File Upload
# ============================================================================

@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a PDF file for rubric generation."""
    lang = get_request_language(request)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=get_message('pdf_only', lang))

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
    request: Request,
    files: List[UploadFile] = File(...),
    batch_id: Optional[str] = None,
    clear: bool = True,
):
    """Accept multiple PDF files, store them, and trigger async ontology extraction."""
    lang = get_request_language(request)
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
                    reason=get_message('pdf_only', lang),
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
async def get_batch_status(batch_id: str, request: Request):
    """Return extraction status for each document in the batch."""
    lang = get_request_language(request)
    bm = get_batch_manager()
    batch = bm.get_batch_status(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=get_message('batch_not_found', lang))

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

def _concatenate_documents(doc_ids: List[str], max_chars: int = 60000) -> str:
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
async def generate_rubric(gen_request: GenerateRequest, request: Request):
    """Generate a rubric from uploaded PDF document(s)."""
    lang = get_request_language(request)
    language_name = LANGUAGE_NAMES.get(lang, 'español')

    # --- Multi-document path ---
    if gen_request.document_ids:
        document_text = _concatenate_documents(gen_request.document_ids)
        logger.info(
            f"📄 Concatenated {len(gen_request.document_ids)} documents, "
            f"{len(document_text)} chars"
        )
    else:
        # --- Single-document backward-compat path ---
        pdf_path = UPLOAD_DIR / f"{gen_request.document_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=get_message('doc_not_found', lang))

        document_text = extract_text_from_pdf(str(pdf_path))
        if not document_text.strip():
            raise HTTPException(
                status_code=400,
                detail=get_message('no_text_extracted', lang),
            )
        logger.info(f"📄 Extracted {len(document_text)} chars from PDF")
        document_text = document_text[:60000]

    # --- Semantic search for similar rubrics ---
    if not gen_request.skip_search:
        try:
            similar = _get_rubric_repository_service().search_similar(document_text[:3000])
            if similar:
                # Find matching local repo filenames
                repo_files = {f.stem: f.name for f in RUBRICS_REPO_DIR.glob("*.docx")}
                similar_names = []
                for s in similar:
                    # Try to find the local file by matching source_filenames or rubric_id via .meta
                    matched_name = None
                    for meta_file in RUBRICS_REPO_DIR.glob("*.meta"):
                        try:
                            meta_id = meta_file.read_text(encoding="utf-8").strip()
                            if meta_id == s.get("rubric_id", ""):
                                docx_name = meta_file.stem + ".docx"
                                if (RUBRICS_REPO_DIR / docx_name).exists():
                                    matched_name = docx_name
                                    break
                        except:
                            pass
                    similar_names.append({
                        "filename": matched_name or "rúbrica sin archivo",
                        "score": round((s.get("score", 0) * 100)),
                        "rubric_id": s.get("rubric_id", ""),
                    })

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
                    "similar_names": similar_names,
                }
        except Exception as e:
            logger.warning(f"⚠️ Semantic search failed, proceeding with generation: {e}")

    # --- Base rubric support ---
    base_rubric_section = ""
    if gen_request.base_rubric_id:
        base_rubric = _get_rubric_repository_service().get_rubric(gen_request.base_rubric_id)
        if not base_rubric:
            raise HTTPException(status_code=404, detail="Rúbrica base no encontrada")
        base_rubric_text = base_rubric.get("rubric_text", "")
        base_rubric_section = (
            f"\n\n--- RÚBRICA BASE DE REFERENCIA ---\n{base_rubric_text}\n\n"
            f"Usa esta rúbrica como base y adaptala al nuevo documento."
        )

    # ---- STEP 1: Extract specific requirements from the document via LLM ----
    institution = os.getenv('INSTITUTION_NAME', '')
    topics_list = []
    try:
        import litellm as _litellm

        topic_response = _litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Eres un analizador de documentos normativos. "
                    "Lee el documento y extrae TODOS los requisitos, obligaciones y aspectos evaluables que contiene. "
                    "Cada requisito debe ser específico (no genérico). "
                    "Por ejemplo, si el documento habla de dispensa académica y menciona que hay que incluir "
                    "metodoloxías susceptibles de dispensa Y un sistema de avaliación alternativo, "
                    "esos son DOS requisitos separados.\n\n"
                    "Responde SOLO con un JSON array de strings. Cada string es un requisito específico "
                    "tal como aparece en el documento.\n"
                    "Ejemplo: [\n"
                    "  \"Incluír relación de metodoloxías susceptibles de dispensa académica (apartado 5.4)\",\n"
                    "  \"Incluír sistema de avaliación para estudantado con dispensa concedida (artigo 14.5)\",\n"
                    "  \"Usar fórmulas inclusivas de referirse ás persoas nas guías docentes\"\n"
                    "]"
                )},
                {"role": "user", "content": f"Extrae todos los requisitos de este documento:\n\n{document_text[:50000]}"},
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        raw_topics = topic_response.choices[0].message.content or "[]"
        logger.info(f"📋 Raw requirements response: {raw_topics[:500]}")
        # Parse JSON
        raw_topics = raw_topics.strip()
        if raw_topics.startswith("```"):
            raw_topics = raw_topics.split("\n", 1)[-1]
        if raw_topics.endswith("```"):
            raw_topics = raw_topics.rsplit("```", 1)[0]
        raw_topics = raw_topics.strip()
        try:
            topics_list = json.loads(raw_topics)
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Could not parse requirements JSON: {raw_topics[:200]}")
            topics_list = []
        
        if not isinstance(topics_list, list):
            topics_list = []
        
        logger.info(f"📋 Extracted {len(topics_list)} requirements: {topics_list}")
    except Exception as e:
        logger.warning(f"⚠️ Requirement extraction failed: {e}")
        topics_list = []

    # ---- STEP 2: Generate rubric ----
    # Map level to percentage and description
    _LEVEL_INFO = {
        "inicial": ("60%", "Operacional (Básico)"),
        "avanzado": ("80%", "Técnico/Regulatorio (Intermedio)"),
        "critico": ("95%", "Alta Criticidad (Legal)"),
    }
    level_pct, level_desc = _LEVEL_INFO.get(gen_request.level, ("80%", "Técnico/Regulatorio"))

    if topics_list:
        # Use extracted requirements to guide generation
        numbered_topics = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics_list))
        topic_count = len(topics_list)
        agent_message = (
            f"[IDIOMA: Toda tu respuesta DEBE estar en {language_name}.]\n\n"
            f"Institución: {institution}\n"
            f"Nivel de exigencia: {level_desc} — cumplimiento mínimo requerido: {level_pct}\n\n"
            f"Estos son los {topic_count} requisitos extraídos del documento. "
            f"Genera una fila en la tabla para cada uno:\n\n"
            f"{numbered_topics}\n\n"
            f"--- TEXTO DEL DOCUMENTO NORMATIVO ---\n{document_text}"
            f"{base_rubric_section}"
        )
    else:
        # Fallback: direct generation without requirement extraction
        logger.warning("⚠️ No requirements extracted, falling back to direct generation")
        agent_message = (
            f"[IDIOMA: Toda tu respuesta DEBE estar en {language_name}.]\n\n"
            f"Institución: {institution}\n"
            f"Nivel de exigencia: {level_desc} — cumplimiento mínimo requerido: {level_pct}\n\n"
            f"Lee el documento y genera una rúbrica con un criterio por cada requisito que encuentres.\n\n"
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
            generate_content_config=types.GenerateContentConfig(temperature=0.0),
            instruction=(
                "Eres un especialista en compliance y auditoría normativa.\n\n"
                "Tu tarea: recibir una lista de tópicos extraídos de un documento normativo y generar "
                "una rúbrica de cumplimiento con un criterio por cada tópico.\n\n"
                "### Reglas\n"
                "- Genera EXACTAMENTE una fila en la tabla por cada tópico de la lista.\n"
                "- Usa el texto del documento para formular cada criterio con detalle.\n"
                "- No agregues filas para temas que no estén en la lista de tópicos.\n"
                "- No omitas ningún tópico de la lista.\n"
                "- Evita términos sexistas. Usa lenguaje respetuoso con la igualdad de género.\n"
                "- No uses términos vagos sin definirlos.\n\n"
                "### Nivel de exigencia y su efecto en los criterios\n"
                "El nivel de exigencia indicado en el mensaje determina la rigurosidad de los umbrales:\n"
                "- Operacional (60%): umbrales mínimos, basta con cumplir lo básico. Ejemplo: 'Mencionar al menos 1 método'.\n"
                "- Técnico/Regulatorio (80%): umbrales moderados, cumplimiento sustancial. Ejemplo: 'Describir al menos 2 métodos con detalle'.\n"
                "- Alta Criticidad (95%): umbrales estrictos, cumplimiento casi total. Ejemplo: 'Describir todos los métodos con detalle, justificación y evidencia documental'.\n"
                "A mayor nivel, los umbrales deben ser más exigentes, pedir más cantidad, más detalle y más evidencia.\n\n"
                "### Formato de salida\n\n"
                "Información general con viñetas (•), cada una con su valor concreto:\n"
                "• Ámbito de Aplicación: [qué evalúa esta rúbrica]\n"
                "• Normativa de Referencia: [nombre del documento]\n"
                "• Nivel de Criticidad: [descripción del nivel + porcentaje de cumplimiento mínimo requerido]\n"
                "• Objetivos de la evaluación: [propósito]\n\n"
                "Luego la tabla Markdown con 4 columnas:\n"
                "| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |\n\n"
                "La columna 'Nivel mínimo aprobatorio' incluye: el porcentaje de cumplimiento requerido (según el nivel de exigencia indicado), "
                "un umbral concreto y un ejemplo entre paréntesis. "
                "Formato obligatorio: '[porcentaje]% — Umbral mínimo (Exemplo: caso concreto que cumple)'. "
                "TODAS las filas deben tener porcentaje y ejemplo. Sin ellos, la fila está incompleta.\n\n"
                "Responde solo con la rúbrica (viñetas + tabla), sin conversación ni secciones adicionales.\n"
                f"Responde SIEMPRE en {language_name}.\n"
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

        output_filename_txt = f"rubrica_{gen_request.document_id[:8]}.txt"
        output_path_txt = OUTPUT_DIR / output_filename_txt
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(rubric_text)

        # Generate DOCX version
        output_filename_docx = f"rubrica_{gen_request.document_id[:8]}.docx"
        output_path_docx = OUTPUT_DIR / output_filename_docx
        md_to_docx(rubric_text, str(output_path_docx))

        logger.info(f"✅ Rubric generated and saved to DOCX: {output_filename_docx}")

        # Save rubric to local repository folder with descriptive name
        repo_filename = None
        try:
            doc_ids = gen_request.document_ids if gen_request.document_ids else [gen_request.document_id]
            source_filenames = [_file_id_to_filename.get(did, f"{did}.pdf") for did in doc_ids]

            # Generate short descriptive name (max 4 words from first source filename)
            base_name = source_filenames[0] if source_filenames else "rubrica"
            base_name = Path(base_name).stem  # remove extension
            words = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ0-9\s]', ' ', base_name).split()
            short_name = "_".join(words[:4]).lower()
            if not short_name:
                short_name = "rubrica"

            # Add timestamp to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            repo_filename = f"{short_name}_{timestamp}.docx"
            repo_path = RUBRICS_REPO_DIR / repo_filename
            md_to_docx(rubric_text, str(repo_path))
            logger.info(f"📁 Rubric saved to repository: {repo_path}")

            # Also store in Qdrant for semantic search
            rubric_id = _get_rubric_repository_service().store_rubric(
                rubric_text, gen_request.level, source_filenames, doc_ids
            )
            # Save rubric_id mapping for Qdrant cleanup on delete
            if rubric_id:
                meta_path = RUBRICS_REPO_DIR / f"{short_name}_{timestamp}.meta"
                meta_path.write_text(rubric_id, encoding="utf-8")
        except Exception as e:
            logger.error(f"⚠️ Failed to store rubric in repository: {e}")

        return {
            "result": rubric_text,
            "download_url": f"/api/download/{output_filename_docx}",
            "similar_rubrics": [],
        }

    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"{get_message('generation_error', lang)}: {str(e)}")


@app.get("/api/download/{filename}")
async def download_file(filename: str, request: Request):
    """Download a generated rubric file."""
    lang = get_request_language(request)
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=get_message('file_not_found', lang))

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

@app.get("/api/rubrics/files")
async def list_rubric_files():
    """List all rubric DOCX files saved in the local repository folder, with topics from Qdrant."""
    files = []
    repo = _get_rubric_repository_service()
    for f in sorted(RUBRICS_REPO_DIR.glob("*.docx"), key=lambda p: p.stat().st_mtime, reverse=True):
        topics = []
        # Try to get topics from Qdrant via .meta file
        meta_path = RUBRICS_REPO_DIR / (f.stem + ".meta")
        if meta_path.exists():
            try:
                rubric_id = meta_path.read_text(encoding="utf-8").strip()
                if rubric_id:
                    rubric_data = repo.get_rubric(rubric_id)
                    if rubric_data:
                        topics = rubric_data.get("topics", [])
            except Exception:
                pass
        files.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "download_url": f"/api/rubrics/files/{f.name}",
            "topics": topics,
        })
    return {"files": files, "total": len(files)}


@app.get("/api/rubrics/files/{filename}")
async def download_rubric_file(filename: str, request: Request):
    """Download a rubric file from the local repository."""
    lang = get_request_language(request)
    file_path = RUBRICS_REPO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=get_message('file_not_found', lang))
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.delete("/api/rubrics/files/{filename}")
async def delete_rubric_file(filename: str, request: Request):
    """Delete a rubric file from the local repository and Qdrant."""
    lang = get_request_language(request)
    file_path = RUBRICS_REPO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=get_message('file_not_found', lang))
    file_path.unlink()
    logger.info(f"🗑️ Rubric file deleted: {filename}")

    # Check for .meta file with Qdrant rubric_id
    meta_filename = Path(filename).stem + ".meta"
    meta_path = RUBRICS_REPO_DIR / meta_filename
    if meta_path.exists():
        try:
            rubric_id = meta_path.read_text(encoding="utf-8").strip()
            if rubric_id:
                _get_rubric_repository_service().delete_rubric(rubric_id)
                logger.info(f"🗑️ Also removed from Qdrant: {rubric_id}")
            meta_path.unlink()
        except Exception as e:
            logger.warning(f"⚠️ Could not remove from Qdrant: {e}")

    return {"message": f"Archivo {filename} eliminado"}


@app.post("/api/rubrics/files/{filename}/replace")
async def replace_rubric_file(filename: str, request: Request, file: UploadFile = File(...)):
    """Replace a rubric file in the repo with an edited version. Deletes old from Qdrant and stores new."""
    lang = get_request_language(request)
    old_path = RUBRICS_REPO_DIR / filename
    if not old_path.exists():
        raise HTTPException(status_code=404, detail=get_message('file_not_found', lang))

    # Delete old from Qdrant
    meta_filename = Path(filename).stem + ".meta"
    meta_path = RUBRICS_REPO_DIR / meta_filename
    if meta_path.exists():
        try:
            old_rubric_id = meta_path.read_text(encoding="utf-8").strip()
            if old_rubric_id:
                _get_rubric_repository_service().delete_rubric(old_rubric_id)
                logger.info(f"🗑️ Old rubric removed from Qdrant: {old_rubric_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not remove old rubric from Qdrant: {e}")

    # Save new file (overwrite)
    content = await file.read()
    old_path.write_bytes(content)
    logger.info(f"📁 Rubric file replaced: {filename}")

    # Extract text from the new DOCX and store in Qdrant
    try:
        rubric_text = extract_text_from_docx(str(old_path))
        if rubric_text.strip():
            new_rubric_id = _get_rubric_repository_service().store_rubric(
                rubric_text, "avanzado", [filename], []
            )
            if new_rubric_id:
                meta_path.write_text(new_rubric_id, encoding="utf-8")
                logger.info(f"✅ New rubric stored in Qdrant: {new_rubric_id}")
    except Exception as e:
        logger.error(f"⚠️ Failed to store replaced rubric in Qdrant: {e}")

    return {
        "message": f"Rúbrica {filename} reemplazada exitosamente",
        "filename": filename,
        "download_url": f"/api/rubrics/files/{filename}",
    }



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
async def upload_rubric(request: Request, file: UploadFile = File(...)):
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
async def upload_doc(request: Request, file: UploadFile = File(...)):
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
async def run_evaluation(eval_request: EvaluateRequest, request: Request):
    """Run evaluation: compare a document against a rubric."""
    lang = get_request_language(request)
    language_name = LANGUAGE_NAMES.get(lang, 'español')

    # Resolve rubric: from repo filename or uploaded file
    rubric_path = None
    if eval_request.rubric_filename:
        # Use rubric from local repository
        rubric_path = RUBRICS_REPO_DIR / eval_request.rubric_filename
        if not rubric_path.exists():
            raise HTTPException(status_code=404, detail=get_message('rubric_not_found', lang))
    elif eval_request.rubric_id:
        for ext in [".pdf", ".txt", ".md", ".docx"]:
            candidate = UPLOAD_DIR / f"rubric_{eval_request.rubric_id}{ext}"
            if candidate.exists():
                rubric_path = candidate
                break

    if not rubric_path:
        raise HTTPException(status_code=404, detail=get_message('rubric_not_found', lang))

    doc_path = None
    for ext in [".pdf", ".docx"]:
        candidate = UPLOAD_DIR / f"doc_{eval_request.doc_id}{ext}"
        if candidate.exists():
            doc_path = candidate
            break

    if not doc_path:
        raise HTTPException(status_code=404, detail=get_message('doc_not_found', lang))

    if rubric_path.suffix.lower() == ".pdf":
        rubric_text = extract_text_from_pdf(str(rubric_path))
        if not rubric_text.strip():
            raise HTTPException(status_code=400, detail=get_message('no_text_extracted', lang))
    elif rubric_path.suffix.lower() == ".docx":
        rubric_text = extract_text_from_docx(str(rubric_path))
        if not rubric_text.strip():
            raise HTTPException(status_code=400, detail=get_message('no_text_extracted', lang))
    else:
        rubric_text = rubric_path.read_text(encoding="utf-8")

    if doc_path.suffix.lower() == ".docx":
        document_text = extract_text_from_docx(str(doc_path))
    else:
        document_text = extract_text_from_pdf(str(doc_path))
    if not document_text.strip():
        raise HTTPException(status_code=400, detail=get_message('no_text_extracted', lang))

    logger.info(f"📋 Evaluating: rubric={len(rubric_text)} chars, doc={len(document_text)} chars")

    # Two-step compatibility check: 1) Same institution, 2) Semantic topic similarity
    if eval_request.rubric_filename:
        try:
            import litellm as _litellm
            import json as _json

            # Get configured institution from env
            configured_institution = os.getenv("INSTITUTION_NAME", "")

            meta_path = RUBRICS_REPO_DIR / (Path(eval_request.rubric_filename).stem + ".meta")
            rubric_data = None
            if meta_path.exists():
                rubric_id_qdrant = meta_path.read_text(encoding="utf-8").strip()
                rubric_data = _get_rubric_repository_service().get_rubric(rubric_id_qdrant)

            if rubric_data:
                source_filenames = rubric_data.get("source_filenames", [])
                topics = rubric_data.get("topics", [])
                rubric_summary = rubric_data.get("summary", "")[:500]

                # STEP 1: Check document belongs to configured institution
                if configured_institution:
                    inst_response = _litellm.completion(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": (
                                "Determina si el documento pertenece a la institución indicada. "
                                "Responde SOLO con JSON: {\"belongs\": true/false, \"doc_institution\": \"nombre detectado\"}\n"
                                "Si no puedes determinar la institución del documento, pon belongs: true."
                            )},
                            {"role": "user", "content": (
                                f"Institución esperada: {configured_institution}\n\n"
                                f"Documento:\n{document_text[:1500]}"
                            )},
                        ],
                        temperature=0.0,
                        max_tokens=100,
                    )
                    raw_inst = inst_response.choices[0].message.content or "{}"
                    raw_inst = raw_inst.strip()
                    if raw_inst.startswith("```"): raw_inst = raw_inst.split("\n", 1)[-1]
                    if raw_inst.endswith("```"): raw_inst = raw_inst.rsplit("```", 1)[0]
                    inst = _json.loads(raw_inst.strip())

                    belongs = inst.get("belongs", True)
                    doc_inst = inst.get("doc_institution", "desconocida")

                    logger.info(f"🏛️ Institution check: expected={configured_institution}, doc={doc_inst}, belongs={belongs}")

                    if not belongs:
                        return {
                            "result": get_message('institution_mismatch', lang),
                            "download_url": "",
                            "topic_mismatch": True,
                        }

                # STEP 2: Semantic topic similarity via embeddings
                topics_text = ", ".join(topics) if topics else rubric_summary
                rubric_context = f"Temas: {topics_text}. Fuentes: {', '.join(source_filenames)}"

                qdrant = _get_qdrant_service()
                rubric_vec = qdrant.embed(rubric_context)
                doc_vec = qdrant.embed(document_text[:2000])

                dot_product = sum(a * b for a, b in zip(rubric_vec, doc_vec))
                mag_r = sum(a * a for a in rubric_vec) ** 0.5
                mag_d = sum(a * a for a in doc_vec) ** 0.5
                similarity = dot_product / (mag_r * mag_d) if mag_r and mag_d else 0

                logger.info(f"📊 Topic similarity: {similarity:.2%} | Topics: {topics}")

                if similarity < 0.40:
                    return {
                        "result": get_message('topic_mismatch', lang),
                        "download_url": "",
                        "topic_mismatch": True,
                    }

        except Exception as e:
            logger.warning(f"⚠️ Compatibility check failed, proceeding with evaluation: {e}")

    # Build the evaluation message with inline text
    # Extract the level from the rubric text to guide evaluation strictness
    eval_level = "intermedio"
    eval_level_pct = "80%"
    rubric_lower = rubric_text.lower()
    if "95%" in rubric_lower or "alta criticidad" in rubric_lower or "crítico" in rubric_lower:
        eval_level = "crítico"
        eval_level_pct = "95%"
    elif "60%" in rubric_lower or "operacional" in rubric_lower or "básico" in rubric_lower or "inicial" in rubric_lower:
        eval_level = "inicial"
        eval_level_pct = "60%"

    agent_message = (
        f"[IDIOMA: Responde en {language_name}.]\n\n"
        f"Nivel de exigencia de la rúbrica: {eval_level} (cumplimiento mínimo: {eval_level_pct})\n\n"
        f"IMPORTANTE sobre el nivel de exigencia:\n"
        f"- Si el nivel es 'inicial' (60%): sé FLEXIBLE al evaluar. Si el documento menciona el tema aunque sea brevemente, marca 'Cumple'. "
        f"Solo marca 'No Cumple' si el tema está completamente ausente.\n"
        f"- Si el nivel es 'intermedio' (80%): evalúa con rigor moderado. El documento debe abordar el tema con cierto detalle.\n"
        f"- Si el nivel es 'crítico' (95%): sé MUY ESTRICTO. El documento debe cubrir cada criterio con detalle completo, "
        f"evidencia explícita y sin ambigüedades. Cualquier omisión o vaguedad es 'No Cumple' o 'Parcialmente Cumple'.\n\n"
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
            generate_content_config=types.GenerateContentConfig(temperature=0.0),
            instruction=(
                "Eres un experto en auditoría y cumplimiento normativo. "
                "Tu tarea es evaluar un documento contra una rúbrica de cumplimiento.\n\n"
                "IMPORTANTE: El nivel de exigencia indicado en el mensaje determina tu rigurosidad:\n"
                "- Nivel inicial (60%): sé flexible. Si el documento menciona el tema, marca Cumple.\n"
                "- Nivel intermedio (80%): rigor moderado. El tema debe estar desarrollado.\n"
                "- Nivel crítico (95%): muy estricto. Exige detalle completo y evidencia explícita.\n\n"
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
                "Sé objetivo y profesional. Basa tus comentarios "
                "exclusivamente en la evidencia del documento.\n"
                f"Responde SIEMPRE en {language_name}."
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

        output_filename_txt = f"evaluacion_{eval_request.doc_id[:8]}.txt"
        output_path_txt = OUTPUT_DIR / output_filename_txt
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(eval_text)

        # Generate DOCX version
        output_filename_docx = f"evaluacion_{eval_request.doc_id[:8]}.docx"
        output_path_docx = OUTPUT_DIR / output_filename_docx
        md_to_docx(eval_text, str(output_path_docx))

        logger.info(f"✅ Evaluation completed and saved to DOCX: {output_filename_docx}")

        return {
            "result": eval_text,
            "download_url": f"/api/download/{output_filename_docx}",
        }

    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"{get_message('evaluation_error', lang)}: {str(e)}")


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
# Main
# ============================================================================

def main():
    """Start the server."""
    host = os.getenv("ORCHESTRATOR_HOST", "localhost")
    port = int(os.getenv("ORCHESTRATOR_PORT", "8000"))

    logger.info(f"🚀 Starting AsistIAG server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
