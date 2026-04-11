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

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from common.config import setup_langsmith
from app.main_agent import create_root_agent
from app.qdrant_service import TOOL_REGISTRY
from app.skill_loader import load_skills

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import markdown
from xhtml2pdf import pisa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

UPLOAD_DIR = Path("/tmp/rubricas_uploads")
OUTPUT_DIR = Path("/tmp/rubricas_output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Skills directory
SKILLS_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

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


def md_to_pdf(md_text: str, output_path: str):
    """Convert Markdown text to a PDF file using markdown and xhtml2pdf."""
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    # Enhanced CSS with landscape orientation for tables and better formatting
    html_with_css = f"""
    <html>
    <head>
    <style>
        @page {{
            size: A4 landscape;
            margin: 1.5cm;
        }}
        body {{ 
            font-family: Helvetica, Arial, sans-serif; 
            font-size: 10px;
            color: #333;
            line-height: 1.4;
        }}
        h1 {{ 
            color: #0056b3; 
            font-size: 18px;
            margin-top: 10px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h2 {{ 
            color: #0056b3; 
            font-size: 14px;
            margin-top: 12px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }}
        h3 {{ 
            color: #0056b3; 
            font-size: 12px;
            margin-top: 10px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            margin-bottom: 8px;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin-bottom: 15px;
            margin-top: 10px;
            font-size: 9px;
            page-break-inside: avoid;
        }}
        th, td {{ 
            border: 1px solid #999; 
            padding: 6px 8px; 
            text-align: left;
            vertical-align: top;
            word-wrap: break-word;
        }}
        th {{ 
            background-color: #0056b3; 
            color: white;
            font-weight: bold;
            font-size: 10px;
        }}
        tr:nth-child(even) {{ 
            background-color: #f5f5f5; 
        }}
        tr:nth-child(odd) {{ 
            background-color: #ffffff; 
        }}
        /* Prevent page breaks inside table rows */
        tr {{
            page-break-inside: avoid;
        }}
        /* Better list formatting */
        ul, ol {{
            margin-left: 20px;
            margin-bottom: 10px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        /* Code blocks */
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 9px;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    with open(output_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_with_css, dest=pdf_file)
        if pisa_status.err:
            logger.error(f"Error creating PDF: {pisa_status.err}")



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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint — routes through the root agent."""
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    logger.info(f"📩 Chat received: {user_message[:80]}...")

    try:
        response_text = await run_agent(user_message)
    except Exception as e:
        logger.error(f"❌ Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"✅ Agent response: {response_text[:100]}...")

    # Detect action requests for frontend components
    response_type = "text"
    metadata = {"architecture": "skills"}

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
                 "genera una rúbrica", "necesito una rúbrica", "ACTION:GENERATOR"]
    msg_lower = user_msg.lower()
    response_lower = response.lower()
    # Only trigger if keywords are in user message OR explicitly in response
    return any(k in msg_lower for k in keywords) or "ACTION:GENERATOR" in response


def _needs_evaluator_action(user_msg: str, response: str) -> bool:
    """Detect if the chat response should trigger the evaluator UI."""
    keywords = ["evaluar documento", "evaluar un documento", "evaluación",
                 "evaluar cumplimiento", "ACTION:EVALUATOR"]
    msg_lower = user_msg.lower()
    response_lower = response.lower()
    # Only trigger if keywords are in user message OR explicitly in response
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
# API Endpoints - Generator
# ============================================================================

@app.post("/api/generate")
async def generate_rubric(request: GenerateRequest):
    """Generate a rubric from an uploaded PDF document."""
    pdf_path = UPLOAD_DIR / f"{request.document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    document_text = extract_text_from_pdf(str(pdf_path))
    if not document_text.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

    logger.info(f"📄 Extracted {len(document_text)} chars from PDF")

    # Build the message for the generator skill
    agent_message = (
        f"Por favor genera una rúbrica de cumplimiento a partir del siguiente "
        f"documento normativo. Nivel de exigencia: {request.level}.\n\n"
        f"Instrucciones adicionales: {request.prompt}\n\n"
        f"--- DOCUMENTO NORMATIVO ---\n{document_text[:30000]}"
    )

    try:
        rubric_text = await run_agent(agent_message)

        output_filename_txt = f"rubrica_{request.document_id[:8]}.txt"
        output_path_txt = OUTPUT_DIR / output_filename_txt
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(rubric_text)

        # Generate PDF version
        output_filename_pdf = f"rubrica_{request.document_id[:8]}.pdf"
        output_path_pdf = OUTPUT_DIR / output_filename_pdf
        md_to_pdf(rubric_text, str(output_path_pdf))

        logger.info(f"✅ Rubric generated and saved to PDF: {output_filename_pdf}")

        return {
            "result": rubric_text,
            "download_url": f"/api/download/{output_filename_pdf}",
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
    """Upload a rubric file for evaluation."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    if ext not in [".txt", ".md", ".pdf"]:
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
    file_path = UPLOAD_DIR / f"doc_{file_id}.pdf"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"📤 Document uploaded: {file.filename} → doc_{file_id}")
    return {"id": file_id, "filename": file.filename}


@app.post("/api/evaluate/run")
async def run_evaluation(request: EvaluateRequest):
    """Run evaluation: compare a document against a rubric using the evaluador-de-cumplimiento skill."""
    # Verify files exist
    rubric_path = None
    for ext in [".pdf", ".txt", ".md"]:
        candidate = UPLOAD_DIR / f"rubric_{request.rubric_id}{ext}"
        if candidate.exists():
            rubric_path = candidate
            break

    if not rubric_path:
        raise HTTPException(status_code=404, detail="Rúbrica no encontrada")

    doc_path = UPLOAD_DIR / f"doc_{request.doc_id}.pdf"
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    logger.info(f"📋 Starting evaluation via skill: rubric={request.rubric_id}, doc={request.doc_id}")

    # Build the message for the evaluator skill
    # The skill will use its tools to read the files and optionally query Qdrant
    agent_message = (
        f"Evalúa el documento contra la rúbrica de cumplimiento.\n\n"
        f"IDs de archivos:\n"
        f"- rubric_id: {request.rubric_id}\n"
        f"- document_id: {request.doc_id}\n\n"
        f"Usa las herramientas `leer_rubrica_subida` y `leer_documento_subido` para obtener el contenido. "
        f"Si encuentras referencias a normativas externas en el documento, usa `buscar_contexto_qdrant` "
        f"para enriquecer la evaluación con contexto normativo adicional."
    )

    try:
        # Use the root agent runner (which has access to all skills including evaluador-de-cumplimiento)
        eval_text = await run_agent(agent_message)

        if not eval_text or eval_text == "Sin respuesta del agente.":
            raise HTTPException(status_code=500, detail="El evaluador no generó respuesta")

        # Save results
        output_filename_txt = f"evaluacion_{request.doc_id[:8]}.txt"
        output_path_txt = OUTPUT_DIR / output_filename_txt
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(eval_text)

        # Generate PDF version
        output_filename_pdf = f"evaluacion_{request.doc_id[:8]}.pdf"
        output_path_pdf = OUTPUT_DIR / output_filename_pdf
        md_to_pdf(eval_text, str(output_path_pdf))

        logger.info(f"✅ Evaluation completed via skill and saved to PDF: {output_filename_pdf}")

        return {
            "result": eval_text,
            "download_url": f"/api/download/{output_filename_pdf}",
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
                "model": fm.metadata.get("model", getattr(fm, "model", "gemini-2.5-flash")),
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

    logger.info(f"🚀 Starting RubricAI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
