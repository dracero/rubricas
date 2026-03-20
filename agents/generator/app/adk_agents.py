"""
Generator ADK Agents — Proper ADK Multi-Agent Architecture.

Uses google.adk.agents.Agent with model, instruction, tools, and sub_agents.
The ADK framework handles LLM calls internally. We provide tools for Qdrant operations.

Architecture:
    root_agent (orchestrator)
    ├── ontologo_agent  (extracts ontology, saves to Qdrant via tool)
    └── rubricador_agent (searches Qdrant for RAG, generates rubric)
"""

import os
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from google.adk.agents import Agent
from google.genai import types
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


# ============================================================================
# QDRANT SERVICE (Infrastructure — not an Agent)
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
# ADK TOOL FUNCTIONS
# ============================================================================

def guardar_ontologia_en_qdrant(entidades: List[Dict[str, Any]], relaciones: List[Dict[str, Any]] = None) -> str:
    """Saves entities and relations extracted from a normative document to the Qdrant vector database.

    Args:
        entidades: List of entities (concepts, criteria, requirements, roles).
                  Each entity dict needs: nombre, tipo, contexto, propiedades.
        relaciones: Optional list of relations between entities.
                  Each relation dict needs: origen, destino, tipo, propiedades.

    Returns:
        A confirmation message with the number of entities and relations saved.
    """
    try:
        if relaciones is None:
            relaciones = []

        entidades_obj = []
        for e in entidades:
            entidades_obj.append(Entidad(
                nombre=e.get("nombre", "Desconocido"),
                tipo=e.get("tipo", "Desconocido"),
                propiedades=e.get("propiedades", {}),
                contexto=e.get("contexto", ""),
                fecha_creacion=datetime.now().isoformat()
            ))

        relaciones_obj = []
        for r in relaciones:
            relaciones_obj.append(Relacion(
                origen=r.get("origen", "Desconocido"),
                destino=r.get("destino", "Desconocido"),
                tipo=r.get("tipo", "Desconocido"),
                propiedades=r.get("propiedades", {})
            ))

        ontologia = Ontologia(
            entidades=entidades_obj,
            relaciones=relaciones_obj,
            metadata={"source": "adk_ontologo"}
        )

        qdrant = _get_qdrant_service()
        success = qdrant.save_ontology(ontologia)

        if success:
            import time
            logger.info("⏳ Guardrail: Durmiendo 5 segundos para liberar TPM...")
            time.sleep(5)  # Artificial delay to help with Groq TPM limits
            return (
                f"✅ Ontología guardada exitosamente en Qdrant: "
                f"{len(entidades)} entidades, {len(relaciones)} relaciones."
            )
        else:
            return "⚠️ No se pudieron guardar entidades en Qdrant."

    except Exception as e:
        logger.error(f"Error in guardar_ontologia_en_qdrant: {e}")
        return f"❌ Error guardando ontología: {str(e)}"


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
# ADK AGENT FACTORY
# ============================================================================

def create_generator_agent() -> Agent:
    """Creates the root ADK agent with ontólogo and rubricador sub-agents.

    Returns:
        The root Agent instance ready to be used with Runner.
    """
    # Initialize the QdrantService singleton eagerly
    _get_qdrant_service()

    # --- Sub-agent 1: Ontólogo ---
    ontologo_agent = Agent(
        name="ontologo",
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        instruction=(
            "Eres un experto en ontologías educativas.\n\n"
            "Tu tarea es analizar el fragmento de texto normativo que recibes y extraer "
            "una ontología COMPACTA con las entidades y relaciones más importantes.\n\n"
            "REGLAS ESTRICTAS:\n"
            "- Extrae MÁXIMO 5 entidades por fragmento.\n"
            "- Cada entidad tiene: nombre (snake_case), tipo, contexto (máx 80 chars), "
            "propiedades (máx 2 claves simples, sin objetos anidados).\n"
            "- Extrae MÁXIMO 5 relaciones totales.\n"
            "- Cada relación tiene: origen, destino, tipo, propiedades ({}).\n"
            "- Llama a `guardar_ontologia_en_qdrant` UNA SOLA VEZ con todo el resultado.\n"
            "- NO generes texto explicativo. Solo llama a la herramienta y confirma.\n"
            "- NO transfieras al rubricador. Solo guarda y termina."
        ),
        tools=[guardar_ontologia_en_qdrant],
    )

    # --- Sub-agent 2: Rubricador ---
    rubricador_agent = Agent(
        name="rubricador",
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        instruction=(
            "Eres un experto en diseño de instrumentos de evaluación académica.\n\n"
            "Tu tarea:\n"
            "1. Llama a `buscar_contexto_qdrant` con una consulta relevante para obtener "
            "el contexto normativo acumulado.\n"
            "2. Genera una RÚBRICA DE EVALUACIÓN en Markdown con:\n"
            "   - Información general (materia, nivel, objetivos)\n"
            "   - Matriz de evaluación (criterios, escala 1-4, evidencias observables)\n"
            "   - Niveles de dominio con ejemplos\n"
            "   - Requisitos mínimos para aprobar\n\n"
            "REGLAS:\n"
            "- Busca contexto ANTES de generar. Solo una llamada a la herramienta.\n"
            "- Sé concreto. Evita términos vagos sin definición.\n"
            "- Responde solo en español."
        ),
        tools=[buscar_contexto_qdrant],
    )

    # --- Root Agent (Orchestrator) ---
    root_agent = Agent(
        name="generador_rubricas",
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        instruction=(
            "Eres el orquestador del sistema de generación de rúbricas.\n\n"
            "Flujo:\n"
            "1. Si hay un fragmento de documento normativo → transfiere al agente `ontologo`.\n"
            "2. Si es el último fragmento o no hay documento → transfiere al `rubricador`.\n"
            "3. Si el usuario solo pide una rúbrica sin documento → transfiere al `rubricador`.\n\n"
            "Responde siempre en español. No generes la rúbrica tú mismo."
        ),
        tools=[guardar_ontologia_en_qdrant, buscar_contexto_qdrant],
        sub_agents=[ontologo_agent, rubricador_agent],
    )

    return root_agent
