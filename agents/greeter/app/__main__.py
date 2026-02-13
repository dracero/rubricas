"""Entry point for the Greeter A2A agent server."""

import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import GreetingAgent
from app.agent_executor import GreeterAgentExecutor
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10003)
def main(host, port):
    """Start the Greeter A2A agent server."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("❌ GOOGLE_API_KEY not set")
        exit(1)

    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="greeter",
        name="Agente de Bienvenida",
        description=(
            "Saluda a los usuarios, explica las capacidades del sistema RubricAI "
            "(generación y evaluación de rúbricas académicas) y ofrece ayuda inicial."
        ),
        tags=["saludar", "explicar sistema", "charlar", "hola", "buen día"],
        examples=[
            "Hola, ¿qué podés hacer?",
            "Buenos días",
            "¿Para qué sirve este sistema?",
        ],
    )

    agent_host_url = os.getenv("HOST_OVERRIDE") or f"http://{host}:{port}/"

    agent_card = AgentCard(
        name="Agente de Bienvenida - RubricAI",
        description=(
            "Saluda a los usuarios, explica las capacidades del sistema RubricAI "
            "y ofrece ayuda inicial. Usar cuando el usuario saluda o pregunta qué "
            "puede hacer el sistema."
        ),
        url=agent_host_url,
        version="1.0.0",
        default_input_modes=GreetingAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=GreetingAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=GreeterAgentExecutor(api_key=api_key),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    logger.info("=" * 60)
    logger.info("👋 Greeter Agent - A2A Protocol")
    logger.info(f"🚀 Starting on {host}:{port}")
    logger.info(f"🔗 Agent URL: {agent_host_url}")
    logger.info("=" * 60)

    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
