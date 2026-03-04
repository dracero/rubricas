"""
Generator Agents — Migrated to BeeAI Framework and Groq.

Uses beeai_framework to orchestrate:
    ontologo_agent  (extracts ontology, saves to Qdrant via tool)
    rubricador_agent (searches Qdrant for RAG, generates rubric)
"""

import os
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import tool

from sentence_transformers import SentenceTransformer

# Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    QDRANT_AVAILABLE = False

from common.config import ConfiguracionColaba, traceable
from common.llm_factory import create_llm

# Domain imports (data structures & constants)
from .domain import (
    Entidad, Relacion, Ontologia,
    NIVELES_ESTUDIANTE,
    limpiar_json_respuesta, parsear_json_con_fallback,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level singleton for QdrantService (initialized lazily)
# ============================================================================
_qdrant_service: Optional["QdrantService"] = None


def _get_qdrant_service() -> "QdrantService":
    """Get or create the module-level QdrantService singleton."""
    global _qdrant_service
    if _qdrant_service is None:
        config = ConfiguracionColaba()
        _qdrant_service = QdrantService(config)
    return _qdrant_service

def _get_llm(llm_config: dict = None):
    """Create an LLM instance using the factory."""
    config = llm_config or {}
    return create_llm(
        provider=config.get("provider", "groq"),
        model_id=config.get("model_id", ""),
        api_key=config.get("api_key", ""),
    )


# ============================================================================
# QDRANT SERVICE (Infrastructure)
# ============================================================================

class QdrantService:
    """Handles all Qdrant vector DB operations: embeddings, save, search."""

    def __init__(self, config: ConfiguracionColaba):
        self.config = config
        self.collection_name = "rubricas_entidades"

        # Qdrant client
        if config.QDRANT_URL and QDRANT_AVAILABLE:
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )
        else:
            self.client = None
            logger.warning("⚠️ Sin conexión a Qdrant")

        # Embedding model
        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        try:
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL_NAME, device=device
            )
            logger.info(f"✅ Embedding model loaded on: {device}")
        except Exception as e:
            logger.warning(f"⚠️ Fallback to CPU for embeddings: {e}")
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL_NAME, device="cpu"
            )

        self._init_collection()

    def _init_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        if not self.client:
            return
        try:
            collections = self.client.get_collections()
            exists = any(
                c.name == self.collection_name
                for c in collections.collections
            )
            if not exists:
                logger.info(f"📦 Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=384,  # all-MiniLM-L6-v2
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"⚠️ Error initializing Qdrant: {e}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        return self.embedding_model.encode(text).tolist()

    @traceable(name="QdrantService.save_ontology", run_type="chain")
    def save_ontology(self, ontologia: Ontologia) -> bool:
        """Save ontology entities and relations to Qdrant."""
        if not self.client:
            logger.warning("⚠️ No Qdrant client — skipping save")
            return False

        points = []

        # Map relations by source entity
        relations_by_entity = defaultdict(list)
        for rel in ontologia.relaciones:
            relations_by_entity[rel.origen].append(rel.to_dict())

        for entidad in ontologia.entidades:
            point_id = hashlib.md5(entidad.nombre.encode()).hexdigest()
            text_for_embedding = f"{entidad.nombre}: {entidad.contexto}"
            vector = self.embed(text_for_embedding)

            payload = entidad.to_dict()
            payload["relaciones_salientes"] = relations_by_entity[entidad.nombre]

            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"✅ Saved {len(points)} entities to Qdrant")
                return True
            except Exception as e:
                logger.error(f"❌ Error saving to Qdrant: {e}")
                return False
        return False

    @traceable(name="QdrantService.search", run_type="retriever")
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.5) -> List[Dict]:
        """Search for similar entities by vector."""
        if not self.client:
            return []

        vector = self.embed(query)

        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                score_threshold=score_threshold
            )
            hits = result.points if hasattr(result, 'points') else result

            resultados = []
            for hit in hits:
                payload = hit.payload.copy() if hit.payload else {}
                payload['score'] = hit.score
                resultados.append(payload)

            avg_score = (
                sum(r['score'] for r in resultados) / len(resultados)
                if resultados else 0
            )
            logger.info(
                f"📊 Qdrant search: {len(resultados)} hits, "
                f"avg_score: {avg_score:.3f}"
            )
            return resultados
        except Exception as e:
            logger.error(f"⚠️ Qdrant search error: {e}")
            return []


# ============================================================================
# BEEAI TOOLS
# ============================================================================

@tool
def guardar_ontologia_en_qdrant(ontologia_json: str) -> str:
    """Parses an ontology JSON and saves entities/relations to the Qdrant vector database.

    Use this tool after extracting entities and relations from a normative document.
    The JSON must have the structure:
    {
        "entidades": [{"nombre": "...", "tipo": "...", "contexto": "...", "propiedades": {...}}],
        "relaciones": [{"origen": "...", "destino": "...", "tipo": "...", "propiedades": {...}}]
    }

    Args:
        ontologia_json: JSON string with the extracted ontology (entidades + relaciones).

    Returns:
        A confirmation message with the number of entities and relations saved.
    """
    try:
        data = parsear_json_con_fallback(ontologia_json)

        entidades = []
        for e in data.get("entidades", []):
            entidades.append(Entidad(
                nombre=e.get("nombre", "Desconocido"),
                tipo=e.get("tipo", "Desconocido"),
                propiedades=e.get("propiedades", {}),
                contexto=e.get("contexto", ""),
                fecha_creacion=datetime.now().isoformat()
            ))

        relaciones = []
        for r in data.get("relaciones", []):
            relaciones.append(Relacion(
                origen=r.get("origen", "Desconocido"),
                destino=r.get("destino", "Desconocido"),
                tipo=r.get("tipo", "Desconocido"),
                propiedades=r.get("propiedades", {})
            ))

        ontologia = Ontologia(
            entidades=entidades,
            relaciones=relaciones,
            metadata={"source": "adk_ontologo"}
        )

        qdrant = _get_qdrant_service()
        success = qdrant.save_ontology(ontologia)

        if success:
            return (
                f"✅ Ontología guardada exitosamente en Qdrant: "
                f"{len(entidades)} entidades, {len(relaciones)} relaciones."
            )
        else:
            return "⚠️ No se pudieron guardar entidades en Qdrant."

    except Exception as e:
        logger.error(f"Error in guardar_ontologia_en_qdrant: {e}")
        return f"❌ Error guardando ontología: {str(e)}"


@tool
def buscar_contexto_qdrant(query: str) -> str:
    """Searches the Qdrant vector database for normative context relevant to the query.

    Use this tool to retrieve knowledge from previously indexed normative documents.
    Returns entities with their similarity scores, descriptions, and relationships.

    Args:
        query: The search query describing what normative context is needed.

    Returns:
        A formatted string with the relevant normative context found in Qdrant.
    """
    try:
        qdrant = _get_qdrant_service()
        results = qdrant.search(query, limit=10)

        if not results:
            return "No se encontró contexto normativo relevante en la base de conocimiento."

        lines = [f"📚 Contexto normativo encontrado ({len(results)} documentos):\n"]
        for item in results:
            score = item.get('score', 0)
            nombre = item.get('nombre', 'N/A')
            contexto = item.get('contexto', '')[:300]
            lines.append(f"- [{score:.3f}] **{nombre}**: {contexto}")

            # Include outgoing relations
            for rel in item.get('relaciones_salientes', []):
                lines.append(
                    f"  → {rel.get('tipo', '?')} → {rel.get('destino', '?')}"
                )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in buscar_contexto_qdrant: {e}")
        return f"❌ Error buscando contexto: {str(e)}"


# ============================================================================
# AGENT RUNNER FACADE
# ============================================================================

async def run_generator_pipeline(document_text: str, prompt: str, level: str, llm_config: dict = None) -> str:
    """Executes the two-step agent pipeline using BeeAI.
    
    1. Ontologo agent extracts ontology and saves it to Qdrant.
    2. Rubricador agent fetches context from Qdrant and generates the rubric.
    """
    _get_qdrant_service()
    llm = _get_llm(llm_config)

    logger.info("🤖 Starting Groq/BeeAI generator pipeline...")

    # --- Phase 1: Ontologo ---
    if document_text and len(document_text) > 50:
        logger.info("🧠 Phase 1: Extracting ontology from document...")
        
        ontologo_memory = UnconstrainedMemory()
        from beeai_framework.backend.message import SystemMessage
        await ontologo_memory.add(SystemMessage(content=(
            "Eres un EXPERTO EN ONTOLOGÍAS EDUCATIVAS y análisis normativo.\n\n"
            "Tu tarea es:\n"
            "1. Analizar el texto normativo proporcionado por el usuario.\n"
            "2. Extraer una ontología con ENTIDADES (conceptos, requisitos) "
            "y RELACIONES (REQUIERE, COMPLEMENTA, REGULA).\n"
            "3. Usar SÍ O SÍ la herramienta `guardar_ontologia_en_qdrant` para persistir la ontología.\n\n"
            "El argumento `ontologia_json` debe ser un string JSON literario con la estructura:\n"
            '{"entidades": [...], "relaciones": [...]}\n\n'
            "NO incluyas explicaciones largas, simplemente extrae y guarda."
        )))

        ontologo_agent = ReActAgent(
            llm=llm,
            tools=[guardar_ontologia_en_qdrant],
            memory=ontologo_memory,
        )

        ontologo_prompt = f"Extrae la ontología y guárdala para el siguiente texto:\n\n{document_text[:20000]}"
        try:
            await ontologo_agent.run(ontologo_prompt)
            logger.info("✅ Phase 1 complete.")
        except Exception as e:
            logger.warning(f"⚠️ Phase 1 (Ontologo) failed or timed out, continuing... {e}")

    # --- Phase 2: Rubricador ---
    logger.info("🧠 Phase 2: Generating Rubric via RAG...")
    
    rubricador_memory = UnconstrainedMemory()
    await rubricador_memory.add(SystemMessage(content=(
        "Eres un ARQUITECTO PEDAGÓGICO experto en diseño de instrumentos de evaluación.\n\n"
        "Tu tarea es:\n"
        "1. Usar la herramienta `buscar_contexto_qdrant` para obtener contexto.\n"
        "2. Generar una RÚBRICA DE EVALUACIÓN detallada.\n\n"
        "ESTRUCTURA OBLIGATORIA de la rúbrica:\n"
        "1. INFORMACIÓN GENERAL (Materia, Nivel, Objetivos)\n"
        "2. COMPETENCIAS A EVALUAR\n"
        "3. MATRIZ DE EVALUACIÓN (Dimensiones, Criterios, Escala 1-4, Evidencias)\n"
        "4. NIVELES DE DOMINIO con ejemplos específicos\n"
        "5. RECOMENDACIONES AL ESTUDIANTE\n\n"
        "Tu respuesta FINAL debe ser EXCLUSIVAMENTE la rúbrica formateada en Markdown."
    )))

    rubricador_agent = ReActAgent(
        llm=llm,
        tools=[buscar_contexto_qdrant],
        memory=rubricador_memory,
    )

    rubricador_prompt = (
        f"NIVEL EDUCATIVO: {level}\n"
        f"SOLICITUD: {prompt}\n"
        f"Por favor, genera la rúbrica ahora."
    )

    try:
        response = await rubricador_agent.run(rubricador_prompt)
        if hasattr(response, 'result') and hasattr(response.result, 'text'):
            return response.result.text
        elif hasattr(response, 'last_message') and response.last_message:
            content = response.last_message.content
            if isinstance(content, list):
                return "".join([c.text for c in content if hasattr(c, "text")])
            return str(content)
        return str(response)
    except Exception as e:
        logger.error(f"❌ Phase 2 (Rubricador) failed: {e}")
        return f"Error al generar la rúbrica: {str(e)}"
