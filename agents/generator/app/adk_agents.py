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
        model="gemini-2.5-flash",
        instruction=(
            "Eres un EXPERTO EN ONTOLOGÍAS EDUCATIVAS y análisis normativo.\n\n"
            "Tu tarea es:\n"
            "1. Analizar el texto normativo proporcionado por el usuario.\n"
            "2. Extraer una ontología con ENTIDADES (conceptos, criterios, requisitos, roles) "
            "y RELACIONES (REQUIERE, COMPLEMENTA, DEFINE, ES_PARTE_DE, REGULA).\n"
            "3. Usar la herramienta `guardar_ontologia_en_qdrant` para persistir la ontología.\n\n"
            "El JSON de ontología debe tener esta estructura:\n"
            '{\n'
            '  "entidades": [\n'
            '    {"nombre": "id_unico", "tipo": "concepto|criterio|requisito|rol", '
            '"contexto": "definición breve", "propiedades": {}}\n'
            '  ],\n'
            '  "relaciones": [\n'
            '    {"origen": "id_1", "destino": "id_2", "tipo": "REQUIERE|ES_PARTE_DE|REGULA", '
            '"propiedades": {}}\n'
            '  ]\n'
            '}\n\n'
            "REGLAS:\n"
            "- Extrae MÍNIMO 5 entidades por documento.\n"
            "- Genera MÍNIMO 3 relaciones por entidad.\n"
            "- Normaliza nombres en snake_case.\n"
            "- Conecta densamente los conceptos.\n"
            "- SIEMPRE usa la herramienta para guardar el resultado.\n"
            "- Cuando termines, transfiere al agente `rubricador` para que genere la rúbrica."
        ),
        tools=[guardar_ontologia_en_qdrant],
    )

    # --- Sub-agent 2: Rubricador ---
    rubricador_agent = Agent(
        name="rubricador",
        model="gemini-2.5-flash",
        instruction=(
            "Eres un ARQUITECTO PEDAGÓGICO experto en diseño de instrumentos de evaluación.\n\n"
            "Tu tarea es:\n"
            "1. Usar la herramienta `buscar_contexto_qdrant` para obtener contexto normativo "
            "relevante de la base de conocimiento.\n"
            "2. Generar una RÚBRICA DE EVALUACIÓN detallada basada en ese contexto.\n\n"
            "ESTRUCTURA OBLIGATORIA de la rúbrica:\n"
            "1. INFORMACIÓN GENERAL (Materia, Nivel, Objetivos)\n"
            "2. COMPETENCIAS A EVALUAR (Cognitivas, Procedimentales, Actitudinales)\n"
            "3. MATRIZ DE EVALUACIÓN (Dimensiones, Criterios, Escala 1-4, Evidencias observables)\n"
            "4. NIVELES DE DOMINIO con ejemplos específicos\n"
            "5. RECOMENDACIONES AL ESTUDIANTE\n\n"
            "REGLAS CRÍTICAS:\n"
            "- NO uses términos vagos como 'efectivo' o 'adecuado' sin definirlos.\n"
            "- Cada criterio debe tener EVIDENCIAS OBSERVABLES.\n"
            "- Incluye REQUISITOS MÍNIMOS concretos para aprobar.\n"
            "- Usa formato Markdown.\n"
            "- SIEMPRE busca contexto en Qdrant ANTES de generar la rúbrica."
        ),
        tools=[buscar_contexto_qdrant],
    )

    # --- Root Agent (Orchestrator) ---
    root_agent = Agent(
        name="generador_rubricas",
        model="gemini-2.5-flash",
        instruction=(
            "Eres el orquestador del sistema de generación de rúbricas académicas.\n\n"
            "Tu flujo de trabajo es:\n"
            "1. Si el usuario proporciona un documento normativo, transfiere al agente "
            "`ontologo` para que extraiga la ontología y la guarde en Qdrant.\n"
            "2. Luego transfiere al agente `rubricador` para que busque contexto en Qdrant "
            "y genere la rúbrica detallada.\n"
            "3. Si el usuario solo pide una rúbrica sin documento, transfiere directamente "
            "al `rubricador` para que use el contexto ya existente en Qdrant.\n\n"
            "IMPORTANTE:\n"
            "- Siempre responde en español.\n"
            "- Si se indica un nivel educativo (inicial, avanzado, posgrado), "
            "inclúyelo en la solicitud al rubricador.\n"
            "- No generes la rúbrica tú mismo, delega siempre al rubricador."
        ),
        sub_agents=[ontologo_agent, rubricador_agent],
    )

    return root_agent
