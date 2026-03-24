"""
Main Agent — Single root ADK agent that loads skills as sub-agents.

Replaces the multi-process architecture with a single agent that
dynamically loads skills from .md files.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent

from app.skill_loader import load_skills
from app.qdrant_service import TOOL_REGISTRY

logger = logging.getLogger(__name__)

# Default skills directory (relative to project root)
DEFAULT_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "skills"
)


def create_root_agent(skills_dir: Optional[str] = None) -> Agent:
    """Create the root ADK agent with skills loaded as sub-agents.

    Args:
        skills_dir: Path to skills directory. Defaults to ./skills/

    Returns:
        Root Agent with all skills loaded as sub_agents.
    """
    skills_path = skills_dir or DEFAULT_SKILLS_DIR

    logger.info("=" * 60)
    logger.info("🚀 Creating Root Agent with Skills")
    logger.info(f"   Skills directory: {skills_path}")
    logger.info(f"   Available tools: {list(TOOL_REGISTRY.keys())}")
    logger.info("=" * 60)

    # Load all skills from the directory
    sub_agents = load_skills(skills_path, TOOL_REGISTRY)

    if not sub_agents:
        logger.info("📭 No skills loaded yet — upload skills via the frontend")

    # List loaded skills for logging
    skill_names = [a.name for a in sub_agents]
    logger.info(f"📋 Loaded skills: {skill_names}")

    # Build dynamic routing instructions from loaded skills
    if sub_agents:
        agents_desc = _build_agents_description(sub_agents)
        routing_instructions = (
            "AGENTES DISPONIBLES:\n"
            f"{agents_desc}\n\n"
            "INSTRUCCIONES:\n"
            "1. Analiza el mensaje del usuario para determinar qué necesita.\n"
            "2. Transfiere al agente más adecuado según la solicitud del usuario.\n"
            "3. Si no estás seguro, pregunta al usuario qué necesita.\n\n"
            "REGLAS:\n"
            "- Siempre responde en español.\n"
            "- No intentes hacer las tareas tú mismo, delega siempre al agente apropiado."
        )
    else:
        routing_instructions = (
            "NOTA: No hay skills cargados actualmente.\n"
            "Informa al usuario que no hay agentes disponibles y que debe subir "
            "skills desde el panel de gestión de Skills en la interfaz.\n"
            "Siempre responde en español."
        )

    # Create root orchestrator agent
    root_agent = Agent(
        name="rubricai_orchestrator",
        model="gemini-2.5-flash",
        instruction=(
            "Eres el orquestador del sistema RubricAI de cumplimiento normativo.\n\n"
            f"{routing_instructions}"
        ),
        sub_agents=sub_agents,
    )

    logger.info("✅ Root agent created successfully")
    return root_agent


def _build_agents_description(agents: list[Agent]) -> str:
    """Build a description of available agents for the root prompt."""
    lines = []
    for agent in agents:
        desc = getattr(agent, 'description', '') or agent.name
        lines.append(f"- `{agent.name}`: {desc}")
    return "\n".join(lines)
