"""Entry point for the Rubric Evaluator A2A agent server."""

import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import RubricEvaluatorAgent
from app.agent_executor import RubricEvaluatorAgentExecutor
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10002)
def main(host, port):
    """Start the Rubric Evaluator A2A agent server."""
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not set")
        exit(1)

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="rubric_evaluator",
        name="Evaluador de Rúbricas",
        description=(
            "Evalúa documentos académicos (PDFs, textos) contra rúbricas "
            "de calidad. Usa RAG con Qdrant para contexto normativo y "
            "genera informes detallados de evaluación con puntajes por criterio."
        ),
        tags=[
            "evaluar",
            "evaluación",
            "calificar",
            "rúbrica",
            "documento",
            "PDF",
            "auditoría",
        ],
        examples=[
            "Evaluá este trabajo de investigación",
            "Quiero evaluar un documento con la rúbrica",
            "Calificar un ensayo académico",
        ],
    )

    agent_host_url = os.getenv("HOST_OVERRIDE") or f"http://{host}:{port}/"

    agent_card = AgentCard(
        name="Evaluador de Rúbricas - RubricAI",
        description=(
            "Evalúa documentos académicos contra rúbricas de calidad usando IA. "
            "Extrae texto de PDFs, consulta normativas en Qdrant, y genera "
            "informes detallados de evaluación con puntajes por criterio."
        ),
        url=agent_host_url,
        version="1.0.0",
        default_input_modes=RubricEvaluatorAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=RubricEvaluatorAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=RubricEvaluatorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info("=" * 60)
    logger.info("📋 Rubric Evaluator Agent - A2A Protocol")
    logger.info(f"🚀 Starting on {host}:{port}")
    logger.info(f"🔗 Agent URL: {agent_host_url}")
    logger.info("=" * 60)

    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
