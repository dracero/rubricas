"""Entry point for the Rubric Generator A2A agent server."""

import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import RubricGeneratorAgent
from app.agent_executor import RubricGeneratorAgentExecutor
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10001)
def main(host, port):
    """Start the Rubric Generator A2A agent server."""
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not set")
        exit(1)

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="rubric_generator",
        name="Generador de Rúbricas",
        description=(
            "Genera rúbricas académicas detalladas basadas en normativas educativas. "
            "Usa RAG con Qdrant para consultar documentos normativos y genera "
            "rúbricas con criterios, niveles de logro e indicadores observables."
        ),
        tags=[
            "generar rúbrica",
            "crear rúbrica",
            "rubric",
            "normativa",
            "criterios",
            "evaluación académica",
        ],
        examples=[
            "Generá una rúbrica para evaluar trabajos de investigación",
            "Necesito una rúbrica para presentaciones orales",
            "Crear rúbrica de evaluación para un ensayo académico",
        ],
    )

    agent_host_url = os.getenv("HOST_OVERRIDE") or f"http://{host}:{port}/"

    agent_card = AgentCard(
        name="Generador de Rúbricas - RubricAI",
        description=(
            "Genera rúbricas académicas detalladas usando IA y normativas educativas "
            "almacenadas en Qdrant. Incluye criterios, niveles de logro, indicadores "
            "observables y requisitos mínimos de calidad."
        ),
        url=agent_host_url,
        version="1.0.0",
        default_input_modes=RubricGeneratorAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=RubricGeneratorAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=RubricGeneratorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info("=" * 60)
    logger.info("📝 Rubric Generator Agent - A2A Protocol")
    logger.info(f"🚀 Starting on {host}:{port}")
    logger.info(f"🔗 Agent URL: {agent_host_url}")
    logger.info("=" * 60)

    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
