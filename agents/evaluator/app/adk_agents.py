"""
Evaluator Agents — Migrated to BeeAI Framework and Groq.

Uses beeai_framework to orchestrate:
    evaluator_agent (searches Qdrant for RAG, generates evaluation report)
"""

import os
import logging
from typing import Dict, List, Any, Optional

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
# BEEAI TOOLS
# ============================================================================

@tool
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
# AGENT RUNNER FACADE
# ============================================================================

async def run_evaluator_pipeline(rubric_text: str, document_text: str, llm_config: dict = None) -> str:
    """Executes the evaluator agent pipeline using BeeAI."""
    _get_qdrant_service()
    llm = _get_llm(llm_config)

    logger.info("🤖 Starting Groq/BeeAI evaluator pipeline...")

    evaluator_memory = UnconstrainedMemory()
    from beeai_framework.backend.message import SystemMessage
    await evaluator_memory.add(SystemMessage(content=(
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
        "- Basa tus juicios SOLO en la evidencia presentada y la rúbrica.\n"
        "Tu respuesta FINAL debe ser EXCLUSIVAMENTE el informe de evaluación en Markdown."
    )))

    evaluator_agent = ReActAgent(
        llm=llm,
        tools=[buscar_contexto_para_evaluacion],
        memory=evaluator_memory,
    )

    evaluator_prompt = (
        f"Por favor evalúa el siguiente documento usando la rúbrica proporcionada.\n\n"
        f"RÚBRICA DE REFERENCIA:\n"
        f"{rubric_text[:10000]}\n\n"
        f"DOCUMENTO DEL ESTUDIANTE:\n"
        f"{document_text[:20000]}\n\n"
        f"Instrucciones adicionales: Busca contexto normativo en Qdrant si es necesario "
        f"para validar los criterios de la rúbrica."
    )

    try:
        response = await evaluator_agent.run(evaluator_prompt)
        if hasattr(response, 'result') and hasattr(response.result, 'text'):
            return response.result.text
        elif hasattr(response, 'last_message') and response.last_message:
            content = response.last_message.content
            if isinstance(content, list):
                return "".join([c.text for c in content if hasattr(c, "text")])
            return str(content)
        return str(response)
    except Exception as e:
        logger.error(f"❌ Evaluator agent failed: {e}")
        return f"Error al generar la evaluación: {str(e)}"
