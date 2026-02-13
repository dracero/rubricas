"""
Evaluator ADK Agents — Proper ADK Multi-Agent Architecture.

Uses google.adk.agents.Agent with model, instruction, tools, and sub_agents.
The ADK framework handles LLM calls internally. We provide tools for Qdrant operations.

Architecture:
    root_agent (evaluator)
    └── uses `buscar_contexto_para_evaluacion` tool to enrich rubric with normative context
"""

import os
import logging
from typing import Dict, List, Any, Optional

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

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        return self.embedding_model.encode(text).tolist()

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

def buscar_contexto_para_evaluacion(consulta: str) -> str:
    """Busca contexto normativo relevante en la base de conocimientos (Qdrant).

    Usa esta herramienta cuando necesites verificar si los criterios de la rúbrica
    están alineados con la normativa institucional o para enriquecer la evaluación.

    Args:
        consulta: Texto que describe el tema o los criterios a verificar.

    Returns:
        String con el contexto normativo encontrado (entidades, definiciones, relaciones).
    """
    try:
        qdrant = _get_qdrant_service()
        # Usamos un límite razonable para no saturar el contexto
        results = qdrant.search(consulta, limit=5, score_threshold=0.4)

        if not results:
            return "No se encontró contexto normativo relevante para esta consulta."

        lines = [f"📚 Contexto normativo encontrado ({len(results)} registros):\n"]
        for item in results:
            score = item.get('score', 0)
            nombre = item.get('nombre', 'N/A')
            contexto = item.get('contexto', '')[:300]
            lines.append(f"- [{score:.2f}] **{nombre}**: {contexto}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in buscar_contexto_para_evaluacion: {e}")
        return f"❌ Error buscando contexto: {str(e)}"


# ============================================================================
# ADK AGENT FACTORY
# ============================================================================

def create_evaluator_agent() -> Agent:
    """Creates the root ADK agent for evaluation.

    Returns:
        The Agent instance ready to be used with Runner.
    """
    # Initialize the QdrantService singleton eagerly
    _get_qdrant_service()

    # --- Root Agent (Evaluator) ---
    evaluator_agent = Agent(
        name="evaluador_rubricas",
        model="gemini-2.5-flash",
        instruction=(
            "Eres un AUDITOR ACADÉMICO riguroso y experto en evaluación educativa.\n\n"
            "TU TAREA:\n"
            "Evaluar un trabajo o documento del estudiante contrastándolo con una RÚBRICA "
            "proporcionada y el CONTEXTO NORMATIVO institucional.\n\n"
            "PROCESO:\n"
            "1. Analiza la rúbrica y el documento que se te entregarán.\n"
            "2. USA la herramienta `buscar_contexto_para_evaluacion` para obtener normativas "
            "relacionadas con los temas de la rúbrica (obligatorio si la rúbrica menciona normas).\n"
            "3. Genera un INFORME DE EVALUACIÓN detallado.\n\n"
            "ESTRUCTURA DEL INFORME:\n"
            "- **Resumen General**: Visión global del desempeño.\n"
            "- **Evaluación por Criterio**: Para cada criterio de la rúbrica:\n"
            "  - Calificación/Nivel asignado.\n"
            "  - **Evidencia**: Cita textual o referencia específica del documento del estudiante.\n"
            "  - Justificación basada en la rúbrica y la normativa (si aplica).\n"
            "- **Oportunidades de Mejora**: Consejos concretos para el estudiante.\n"
            "- **Conclusión Final**: Dictamen de aprobación o revisión.\n\n"
            "REGLAS:\n"
            "- Sé objetivo y constructivo.\n"
            "- Si el documento del estudiante es muy corto o irrelevante, indícalo claramente.\n"
            "- Basa tus juicios SOLO en la evidencia presentada y la rúbrica."
        ),
        tools=[buscar_contexto_para_evaluacion],
    )

    return evaluator_agent
