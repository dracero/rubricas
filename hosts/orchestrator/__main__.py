"""Host/Orchestrator - Main entry point for the RubricAI A2A system.

This is the central server that:
1. Discovers remote A2A agents (Generator, Evaluator, Greeter)
2. Routes user messages to the appropriate agent via Gemini
3. Exposes an HTTP API for the frontend to consume
4. Provides file upload/download endpoints for Generator and Evaluator
"""

import asyncio
import json
import logging
import os
import sys
import uuid
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() # Load env vars early

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hosts.orchestrator.remote_agent_connection import RemoteAgentConnection
from hosts.orchestrator.agent_tools import RemoteAgentTool
from hosts.orchestrator.bee_router import BeeRouter

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

AGENT_URLS = {
    "generator": os.getenv("GENERATOR_URL", "http://localhost:10001"),
    "evaluator": os.getenv("EVALUATOR_URL", "http://localhost:10002"),
    "greeter": os.getenv("GREETER_URL", "http://localhost:10003"),
}

# Directories for file management
UPLOAD_DIR = Path("/tmp/rubricas_uploads")
OUTPUT_DIR = Path("/tmp/rubricas_output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="RubricAI Orchestrator",
    description="A2A Host that routes requests to specialized agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
# Global state
agents: dict[str, RemoteAgentConnection] = {}
router: BeeRouter | None = None


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
    document_id: str


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
# Agent Discovery
# ============================================================================


async def discover_agents():
    """Discover all configured A2A agents."""
    global agents, router

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    for name, url in AGENT_URLS.items():
        conn = RemoteAgentConnection(url)
        try:
            await conn.discover()
            agents[name] = conn
            logger.info(f"✅ Discovered agent: {name} at {url}")
        except Exception as e:
            logger.warning(f"⚠️ Could not discover agent '{name}' at {url}: {e}")

    if not agents:
        logger.error("❌ No agents discovered! Make sure agent servers are running.")
    else:
        logger.info(f"🎯 Discovered {len(agents)} agents: {list(agents.keys())}")

    # Create tools for BeeAI router
    tools = []
    for name, conn in agents.items():
        # Create a tool for each discovered agent
        description = conn.get_description() or f"Agent {name}"
        tool = RemoteAgentTool(
            name=f"{name}_tool",
            description=description,
            conn=conn
        )
        tools.append(tool)
        logger.info(f"🛠️ Created tool: {tool.name}")

    if tools:
        router = BeeRouter(tools=tools)
        logger.info("🐝 BeeRouter initialized with BeeAI Framework")
    else:
        logger.warning("⚠️ No tools created, router not initialized")


@app.on_event("startup")
async def startup():
    """Discover agents on server startup."""
    logger.info("=" * 60)
    logger.info("🧠 RubricAI Orchestrator - A2A Host")
    logger.info("=" * 60)
    await discover_agents()


@app.on_event("shutdown")
async def shutdown():
    """Close agent connections."""
    for conn in agents.values():
        await conn.close()


# ============================================================================
# API Endpoints - Basic
# ============================================================================


@app.get("/")
async def root():
    """Health check and system info."""
    return {
        "system": "RubricAI Orchestrator",
        "version": "1.0.0",
        "agents": {
            name: {
                "url": conn.agent_url,
                "name": conn.get_name(),
                "description": conn.get_description(),
            }
            for name, conn in agents.items()
        },
    }


@app.get("/agents")
async def list_agents():
    """List all discovered agents and their capabilities."""
    result = {}
    for name, conn in agents.items():
        result[name] = conn.agent_card or {"status": "not discovered"}
    return result


@app.post("/agents/rediscover")
async def rediscover():
    """Re-discover all agents (useful if an agent was restarted)."""
    await discover_agents()
    return {"status": "ok", "agents": list(agents.keys())}


# ============================================================================
# API Endpoints - Chat (routes to action_request for generator/evaluator)
# ============================================================================


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - routes message to appropriate agent.

    When the user asks to generate a rubric or evaluate a document,
    returns an action_request so the frontend shows the upload component.
    """
    if not router:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    logger.info(f"📩 Chat received: {user_message[:80]}...")

    # Step 1: Use BeeRouter to process the message via ReAct Agent
    try:
        response_text = await router.route(user_message)
    except Exception as e:
        logger.error(f"❌ Router error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"🐝 BeeRouter response: {response_text[:100]}...")

    # Step 2: Check for action requests (Generator/Evaluator)
    if "ACTION:GENERATOR" in response_text:
        # Extract reasoning/content if available, or use default
        content = response_text.replace("ACTION:GENERATOR", "").strip()
        return ChatResponse(
            source="orchestrator",
            target="user",
            type="action_request",
            content=content or "Para generar una rúbrica, necesito que subas un documento normativo (PDF) y selecciones el nivel educativo.",
            metadata={
                "component": "RubricGenerator",
                "routed_to": "generator",
                "framework": "beeai"
            },
        )

    if "ACTION:EVALUATOR" in response_text:
        content = response_text.replace("ACTION:EVALUATOR", "").strip()
        return ChatResponse(
            source="orchestrator",
            target="user",
            type="action_request",
            content=content or "Para evaluar un documento, necesito que subas la rúbrica de referencia y el documento del estudiante.",
            metadata={
                "component": "RubricEvaluator",
                "routed_to": "evaluator",
                "framework": "beeai"
            },
        )

    # Step 3: Default text response (e.g. from Greeter via tool)
    return ChatResponse(
        source="orchestrator",
        target="user",
        type="text",
        content=response_text,
        metadata={
            "routed_to": "unknown", # BeeAI abstracts this
            "framework": "beeai"
        },
    )


# ============================================================================
# API Endpoints - File Upload (shared)
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
# API Endpoints - Generator
# ============================================================================


@app.post("/api/generate")
async def generate_rubric(request: GenerateRequest):
    """Generate a rubric from an uploaded PDF document.

    1. Reads the uploaded PDF
    2. Sends the text + level to the Generator agent via A2A
    3. Saves the result and returns it with a download URL
    """
    # Find the uploaded file
    pdf_path = UPLOAD_DIR / f"{request.document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado. Suba el archivo primero.")

    # Extract text from PDF
    document_text = extract_text_from_pdf(str(pdf_path))
    if not document_text.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

    logger.info(f"📄 Extracted {len(document_text)} chars from PDF")

    # Check if generator agent is available
    if "generator" not in agents:
        raise HTTPException(status_code=503, detail="Agente Generator no disponible")

    # Build the message for the generator agent
    # Include the document text and level in a structured format
    agent_message = json.dumps({
        "type": "generate_rubric",
        "prompt": request.prompt,
        "level": request.level,
        "document_text": document_text[:30000],  # Limit to avoid token overflow
    }, ensure_ascii=False)

    try:
        conn = agents["generator"]
        response = await conn.send_message(agent_message)
        rubric_text = response.get("text", "Error: sin respuesta")

        # Save the rubric to a file for download
        output_filename = f"rubrica_{request.document_id[:8]}.txt"
        output_path = OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rubric_text)

        logger.info(f"✅ Rubric generated and saved: {output_filename}")

        return {
            "result": rubric_text,
            "download_url": f"/api/download/{output_filename}",
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
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/plain",
    )


# ============================================================================
# API Endpoints - Evaluator
# ============================================================================


@app.post("/api/evaluate/upload_rubric")
async def upload_rubric(file: UploadFile = File(...)):
    """Upload a rubric file (.txt/.md) for evaluation."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".txt"
    file_path = UPLOAD_DIR / f"rubric_{file_id}{ext}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📤 Rubric uploaded: {file.filename} → rubric_{file_id}")
    return {"id": file_id, "filename": file.filename}


@app.post("/api/evaluate/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    """Upload a student document (.pdf) for evaluation."""
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"doc_{file_id}.pdf"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📤 Document uploaded: {file.filename} → doc_{file_id}")
    return {"id": file_id, "filename": file.filename}


@app.post("/api/evaluate/run")
async def run_evaluation(request: EvaluateRequest):
    """Run evaluation: compare a document against a rubric.

    1. Reads the uploaded rubric and document
    2. Sends both to the Evaluator agent via A2A
    3. Returns the evaluation result
    """
    # Find the rubric file
    rubric_path = None
    for ext in [".txt", ".md"]:
        candidate = UPLOAD_DIR / f"rubric_{request.rubric_id}{ext}"
        if candidate.exists():
            rubric_path = candidate
            break

    if not rubric_path:
        raise HTTPException(status_code=404, detail="Rúbrica no encontrada")

    # Find the document
    doc_path = UPLOAD_DIR / f"doc_{request.doc_id}.pdf"
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    # Read rubric text
    rubric_text = rubric_path.read_text(encoding="utf-8")

    # Extract text from student document
    document_text = extract_text_from_pdf(str(doc_path))
    if not document_text.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del documento PDF")

    logger.info(f"📋 Evaluating: rubric={len(rubric_text)} chars, doc={len(document_text)} chars")

    # Check if evaluator agent is available
    if "evaluator" not in agents:
        raise HTTPException(status_code=503, detail="Agente Evaluator no disponible")

    # Build the message for the evaluator agent
    agent_message = json.dumps({
        "type": "evaluate_document",
        "rubric_text": rubric_text[:20000],
        "document_text": document_text[:30000],
    }, ensure_ascii=False)

    try:
        conn = agents["evaluator"]
        response = await conn.send_message(agent_message)
        eval_text = response.get("text", "Error: sin respuesta")

        # Save the evaluation result
        output_filename = f"evaluacion_{request.doc_id[:8]}.txt"
        output_path = OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(eval_text)

        return {
            "result": eval_text,
            "download_url": f"/api/download/{output_filename}",
        }

    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluando: {str(e)}")


# ============================================================================
# Main
# ============================================================================


def main():
    """Start the orchestrator server."""
    host = os.getenv("ORCHESTRATOR_HOST", "localhost")
    port = int(os.getenv("ORCHESTRATOR_PORT", "8000"))

    logger.info(f"🚀 Starting orchestrator on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
