"""Entry point for the Rubric Corrector A2A agent server."""

import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import RubricCorrectorAgent
from app.agent_executor import RubricCorrectorAgentExecutor
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10005)
def main(host, port):
    """Start the Rubric Corrector A2A agent server."""
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("❌ GOOGLE_API_KEY not set")
        exit(1)

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="rubric_corrector",
        name="Corrector de Textos",
        description=(
            "Actúa como un coach de escritura. Recibe un pequeño fragmento de texto "
            "del usuario, consulta normativa a través de Qdrant, y sugiere "
            "mejoras conversacionales."
        ),
        tags=[
            "corregir",
            "corrección",
            "mejorar",
            "redacción",
            "texto",
            "coach",
        ],
        examples=[
            "Revisa este párrafo para mi ensayo",
            "¿Cómo suena esta introducción?",
            "Mejorar redacción de este texto",
        ],
    )

    agent_host_url = os.getenv("HOST_OVERRIDE") or f"http://{host}:{port}/"

    agent_card = AgentCard(
        name="Corrector de Textos - RubricAI",
        description=(
            "Revisa y ayuda a redactar partes de un documento en formato sugerencias "
            "dialogadas usando normativa en Qdrant."
        ),
        url=agent_host_url,
        version="1.0.0",
        default_input_modes=RubricCorrectorAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=RubricCorrectorAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=RubricCorrectorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info("=" * 60)
    logger.info("📋 Rubric Corrector Agent - A2A Protocol")
    logger.info(f"🚀 Starting on {host}:{port}")
    logger.info(f"🔗 Agent URL: {agent_host_url}")
    logger.info("=" * 60)

    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
