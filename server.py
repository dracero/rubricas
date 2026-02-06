import os
import shutil
import uuid
import logging
import json
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import a2a_protocol  # Import A2A Protocol definitions


from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Evaluator logic
from rubricador_qdrant_local import (
    AgenteContexto,
    AgenteDocumento,
    AgenteAuditor
)

# Import Generator logic (Restored)
from rubricas_qdrant_local import (
    ConfiguracionColaba,
    AgenteOntologo,
    AgenteRubricador,
    AgentePersistenciaQdrant,
    # AgentePersistenciaQdrant, # Removed duplicate
    PYPDF_AVAILABLE,
    pypdf
)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

from google import genai
from google.genai import types

# Global instances (Enhanced)
config = None
qdrant_agent = None
ontologo = None
rubricador = None
# New Agents
orchestrator_agent = None
eval_contexto = None
eval_documento = None
eval_auditor = None
sys_instances = {}  # Registry of instantiated agents

class AgentOrchestrator:
    def __init__(self, config):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        # Registry is now empty and populated dynamically
        self.agents_registry = {}

    def register_agent(self, agent_instance):
        """Registers a new agent via its card"""
        if hasattr(agent_instance, "get_agent_card"):
            card = agent_instance.get_agent_card()
            agent_id = card.get("id")
            if agent_id:
                self.agents_registry[agent_id] = card
                logger.info(f"✅ Agente registrado: {card['name']} ({agent_id})")
        else:
            logger.warning(f"⚠️ Agente {agent_instance} no tiene 'get_agent_card()'")

    def decide_route(self, user_message: str) -> Dict[str, Any]:
        """Decides which agent should handle the request and returns an actionable response"""
        print(f"🧠 [Orchestrator] Analyzing: {user_message}")
        
        # Dynamic Prompt based on Registry
        agents_desc = ""
        for i, (aid, card) in enumerate(self.agents_registry.items(), 1):
            agents_desc += f"{i}. ID: \"{aid}\"\n   - Descripción: {card['description']}\n"
        
        prompt = f"""
        Eres un ORQUESTADOR DE SISTEMAS DE IA. Tu trabajo es enrutar la solicitud del usuario al agente correcto.
        
        AGENTES DISPONIBLES:
        {agents_desc}
           
        SOLICITUD DEL USUARIO: "{user_message}"
        
        Responde SOLO con un JSON válido:
        {{
            "target_agent": "ID_DEL_AGENTE" | "unknown",
            "confidence": 0.0-1.0,
            "reasoning": "Breve explicación",
            "suggested_response": "Mensaje para el usuario explicando qué se hará"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            
            target = data.get("target_agent", "unknown")
            message = data.get("suggested_response", "")

            # A2A Protocol Response construction
            response_payload = {
                "source": "orchestrator",
                "target": "user",
                "content": message,
                "type": "text",
                "metadata": {
                    "confidence": data.get("confidence", 0),
                    "reasoning": data.get("reasoning", "")
                }
            }

            if target == "generator":
                response_payload["type"] = "action_request"
                response_payload["metadata"]["action"] = "show_component"
                response_payload["metadata"]["component"] = "RubricGenerator"
                if not message:
                    response_payload["content"] = "Entendido. Para generar una rúbrica, necesito que subas el documento normativo."
            
            elif target == "evaluator":
                response_payload["type"] = "action_request"
                response_payload["metadata"]["action"] = "show_component"
                response_payload["metadata"]["component"] = "RubricEvaluator"
                if not message:
                    response_payload["content"] = "Bien. Para evaluar unos apuntes, necesito la rúbrica y el documento del estudiante."
            
            elif target in self.agents_registry:
                # Generic Agent Handler (e.g., Greeter)
                # If we have a direct instance access, we could invoke it here.
                # However, for A2A modularity, we might want to return a specific type.
                # Since we don't have global access to instances inside this method easily without refactoring,
                # let's assume specific "social" agents return text directly OR we invoke them if possible.
                
                # REFACTOR: Accessing the instance to run the logic
                # For this demo, we will check if it's the greeter and run it.
                # In a full system, this would be a dynamic dispatch lookup.
                pass 
                # NOTE: The execution logic should ideally be here if we want the backend to do the work.
                # Let's modify the return to signal the frontend or handle it.
                
                # Dynamic Dispatch Hack for Demo:
                # We need to find the instance. In a real system, orchestrator has a map of {id: instance}
                # For now, we update the response type to 'text' and let the frontend show the message,
                # BUT the user wants the AGENT to generate the message, not the orchestrator's "suggested_response".
                
                # Getting global system instances to find the agent
                sys = get_system()
                if target == "greeter" and "greeter" in sys_instances:
                    agent_instance = sys_instances["greeter"]
                    # Invoke the agent!
                    agent_response = agent_instance.process_request(user_message)
                    response_payload["content"] = agent_response
                else:
                    # Fallback to orchestrator's suggestion
                    response_payload["content"] = message
            
            else:
                response_payload["content"] = message or "No estoy seguro de a qué agente asignar tu solicitud."

            return response_payload

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                "source": "orchestrator", 
                "target": "user", 
                "type": "error", 
                "content": "Error interno en el orquestador."
            }

def get_system():
    global config, qdrant_agent, ontologo, rubricador
    global orchestrator_agent, eval_contexto, eval_documento, eval_auditor
    
    if config is None:
        try:
            config = ConfiguracionColaba()
            
            # Orchestrator (First, so we can register agents)
            orchestrator_agent = AgentOrchestrator(config)

            # Generator System
            qdrant_agent = AgentePersistenciaQdrant(config)
            ontologo = AgenteOntologo(config)
            rubricador = AgenteRubricador(config)
            
            # Register Generator
            orchestrator_agent.register_agent(rubricador)
            
            # Evaluator System
            eval_contexto = AgenteContexto(config)
            eval_documento = AgenteDocumento(config)
            eval_auditor = AgenteAuditor(config)
            
            # Register Evaluator
            orchestrator_agent.register_agent(eval_auditor)

            # Register LangChain Greeter
            from langchain_agent import GreetingAgent
            greeter = GreetingAgent(config.GOOGLE_API_KEY)
            orchestrator_agent.register_agent(greeter)
            
            # Store in system dict for direct access if needed
            sys_instances["greeter"] = greeter
            
            logger.info("✅ System initialized successfully (Generator + Evaluator + Greeter + Orchestrator)")
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise RuntimeError("Failed to initialize system")
    
    return {
        "generator": (qdrant_agent, ontologo, rubricador),
        "evaluator": (eval_contexto, eval_documento, eval_auditor),
        "orchestrator": orchestrator_agent
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    get_system()
    yield
    # Shutdown logic (if needed)

app = FastAPI(title="Rubricas Qdrant API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class GenerateRequest(BaseModel):
    prompt: str
    level: str = "avanzado"
    document_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class EvaluateRequest(BaseModel):
    rubric_id: str
    doc_id: str

# --- ROUTES ---

@app.post("/api/chat")
async def chat_orchestrator(req: ChatRequest):
    sys = get_system()
    orch = sys["orchestrator"]
    decision = orch.decide_route(req.message)
    return decision

# --- GENERATOR ROUTES ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # ... (Existing logic, kept briefly for context, assuming replace replaces implementation)
    # FOR BREVITY in this replace block, I'm pasting the FULL existing logic + imports
    # But since replace_file_content needs exact match, I will rewrite the whole file content in the next step
    # or carefully target blocks.
    # actually, since the file is small (150 lines), I can replace the whole functional part.
    # WAIT: I should be careful. I'll paste the existing generator handlers + new evaluator handlers.
    
    if not PYPDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="pypdf not installed")
    
    upload_id = str(uuid.uuid4())
    file_path = f"temp_{upload_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        text = ""
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid PDF file")
            
        # Store in Qdrant (Trigger Ontologist)
        sys = get_system()
        qdrant = sys["generator"][0]
        ontologo_agent = sys["generator"][1]
        
        logger.info(f"Processing document {upload_id}...")
        ontologia = ontologo_agent.procesar_documento(text)
        success = qdrant.guardar_ontologia(ontologia)
        
        if not success:
            logger.warning("Failed to save ontology to Qdrant")
            
        return {
            "id": upload_id,
            "filename": file.filename,
            "length": len(text),
            "ontology_stats": {
                "entities": len(ontologia.entidades),
                "relations": len(ontologia.relaciones)
            }
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_rubric(req: GenerateRequest):
    sys = get_system()
    qdrant = sys["generator"][0]
    rubricador_agent = sys["generator"][2]
    
    # 1. Search in Qdrant
    logger.info(f"Searching for: {req.prompt}")
    resultados = qdrant.buscar_similares(req.prompt, limit=10)
    
    contexto_rag = {"resultados": resultados}
    
    # 2. Generate Rubric
    logger.info("Generating rubric...")
    rubrica_text = rubricador_agent.generar_rubrica(req.prompt, contexto_rag, req.level)
    
    # 3. Save to file
    output_filename = f"rubrica_{uuid.uuid4()}.md"
    with open(output_filename, "w") as f:
        f.write(rubrica_text)
        
    return {
        "content": rubrica_text,
        "download_url": f"/api/download/{output_filename}"
    }

# --- EVALUATOR ROUTES ---

@app.post("/api/evaluate/upload_rubric")
async def upload_eval_rubric(file: UploadFile = File(...)):
    """Uploads the rubric TXT/MD to be used for evaluation"""
    upload_id = f"rubrica_ref_{uuid.uuid4()}"
    file_path = f"{upload_id}.txt"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"id": file_path, "filename": file.filename}

@app.post("/api/evaluate/upload_doc")
async def upload_eval_doc(file: UploadFile = File(...)):
    """Uploads the student PDF to be evaluated"""
    upload_id = f"doc_stud_{uuid.uuid4()}"
    file_path = f"{upload_id}.pdf"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"id": file_path, "filename": file.filename}

@app.post("/api/evaluate/run")
async def run_evaluation(req: EvaluateRequest):
    sys = get_system()
    ctx_agent = sys["evaluator"][0]
    doc_agent = sys["evaluator"][1]
    aud_agent = sys["evaluator"][2]
    
    logger.info(f"Running evaluation: Rubric={req.rubric_id}, Doc={req.doc_id}")
    
    # 1. Get Knowledge (Rubric + RAG)
    conocimiento = ctx_agent.obtener_conocimiento(req.rubric_id)
    
    # 2. Extract Text from PDF
    texto_pdf = doc_agent.extraer_texto_pdf(req.doc_id)
    
    if len(texto_pdf) < 50:
        raise HTTPException(status_code=400, detail="PDF seems empty or unreadable")
        
    # 3. Audit/Evaluate
    informe = aud_agent.evaluar_documento(conocimiento, texto_pdf)
    
    output_filename = f"evaluacion_{uuid.uuid4()}.md"
    with open(output_filename, "w") as f:
        f.write(informe)
        
    return {
        "content": informe,
        "download_url": f"/api/download/{output_filename}"
    }

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filename, filename=filename)

@app.get("/health")
def health():
    return {"status": "ok"}
