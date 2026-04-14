"""
Skill Loader — Reads .md skill files and creates ADK Agents dynamically.

Each skill file uses YAML frontmatter + Markdown body:
    ---
    name: agent_name
    description: What this agent does
    model: gemini-2.5-flash
    tools:
      - tool_function_name
    sub_agents:
      - sub_agent_name
    ---
    # Markdown instructions for the agent
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.skills import list_skills_in_dir, load_skill_from_dir
from google.adk.skills.models import Skill

logger = logging.getLogger(__name__)


def _parse_sub_agents_from_body(body: str) -> Dict[str, Dict[str, Any]]:
    """Parse sub-agent definitions from the skill body (L2 instructions).

    Sub-agents are defined with:
        ## sub_agent: name
        ### Instrucciones
        ...
        ### Tools
        - tool_name

    Returns:
        Dict mapping sub_agent_name -> {"instruction": str, "tools": list[str]}
    """
    sub_agents = {}

    # Split on ## sub_agent: sections
    parts = re.split(r'^## sub_agent:\s*(\w+)\s*$', body, flags=re.MULTILINE)

    # parts[0] is the main body before any sub_agent section
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        name = parts[i].strip()
        sub_body = parts[i + 1].strip()

        # Extract tools list from ### Tools section
        tools = []
        tools_match = re.search(
            r'^### Tools\s*\n(.*?)(?=^###|\Z)',
            sub_body,
            re.MULTILINE | re.DOTALL
        )
        if tools_match:
            for line in tools_match.group(1).strip().splitlines():
                tool_name = line.strip().lstrip("- ").strip()
                if tool_name:
                    tools.append(tool_name)

        # Extract instruction (everything before ### Tools, excluding ### Instrucciones header)
        instruction = re.sub(
            r'^### Tools\s*\n.*$', '', sub_body,
            flags=re.MULTILINE | re.DOTALL
        ).strip()

        # Remove ### Instrucciones header
        instruction = re.sub(r'^### Instrucciones\s*\n', '', instruction).strip()

        sub_agents[name] = {
            "instruction": instruction,
            "tools": tools,
        }

    return sub_agents


def _get_main_instruction(body: str) -> str:
    """Extract the main instruction from the body (before any sub_agent sections)."""
    parts = re.split(r'^## sub_agent:', body, flags=re.MULTILINE)
    return parts[0].strip()


def load_skills(
    skills_dir: str,
    available_tools: Dict[str, Callable],
    default_model: Any = LiteLlm(model="openai/gpt-4o-mini"),
) -> List[Agent]:
    """Load all skill directories from a directory and create ADK Agents.

    Args:
        skills_dir: Path to directory containing skill folders.
        available_tools: Dict mapping tool function names to actual callables.
        default_model: Default model to use if not specified in skill file.

    Returns:
        List of ADK Agent instances, ready to be used as sub_agents of a root agent.
    """
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        logger.warning(f"⚠️ Skills directory not found: {skills_dir}")
        return []

    agents = []
    
    try:
        skills_dict = list_skills_in_dir(skills_path)
    except Exception as e:
        logger.error(f"Error reading skills directory {skills_dir}: {e}")
        return []

    if not skills_dict:
        logger.info(f"📂 No skill directories found in {skills_dir} (upload skills via frontend)")
        return []

    logger.info(f"📂 Loading {len(skills_dict)} skills from {skills_dir}")

    for skill_name in skills_dict.keys():
        skill_path = skills_path / skill_name
        try:
            skill = load_skill_from_dir(skill_path)
            agent = _create_agent_from_skill(skill, available_tools, default_model)
            agents.append(agent)
            logger.info(f"  ✅ Loaded skill: {agent.name} ({skill_name})")
        except Exception as e:
            logger.error(f"  ❌ Error loading skill {skill_name}: {e}")

    logger.info(f"🎯 Loaded {len(agents)} skills total")
    return agents


def _create_agent_from_skill(
    skill: Skill,
    available_tools: Dict[str, Callable],
    default_model: str,
) -> Agent:
    """Load an ADK Skill object and create an ADK Agent."""
    
    fm = skill.frontmatter
    
    # Retrieve ad-hoc metadata fields
    extra_tools = fm.metadata.get("tools", getattr(fm, "tools", []))
    extra_sub_agents = fm.metadata.get("sub_agents", getattr(fm, "sub_agents", []))
    model = fm.metadata.get("model", getattr(fm, "model", default_model))

    # If model is a string (from SKILL.md frontmatter), wrap it in LiteLlm
    if isinstance(model, str):
        model = LiteLlm(model=model)

    name = skill.name.replace("-", "_")

    # Resolve tool functions
    tools = []
    for tool_name in extra_tools:
        if tool_name in available_tools:
            tools.append(available_tools[tool_name])
        else:
            logger.warning(f"  ⚠️ Tool '{tool_name}' not found for skill '{name}'")

    # Parse sub-agents from L2 instructions
    sub_agents_defs = _parse_sub_agents_from_body(skill.instructions)

    # Create sub-agent instances
    sub_agents = []
    for sub_name_raw in extra_sub_agents:
        if sub_name_raw in sub_agents_defs:
            sub_def = sub_agents_defs[sub_name_raw]
            
            # Sanitize sub-agent name
            sub_name = sub_name_raw.replace("-", "_")

            # Resolve sub-agent tools
            sub_tools = []
            for t_name in sub_def["tools"]:
                if t_name in available_tools:
                    sub_tools.append(available_tools[t_name])

            sub_agent = Agent(
                name=sub_name,
                model=model,
                instruction=sub_def["instruction"],
                tools=sub_tools if sub_tools else None,
            )
            sub_agents.append(sub_agent)
            logger.info(f"    ↳ Sub-agent: {sub_name} (tools: {sub_def['tools']})")
        else:
            logger.warning(f"  ⚠️ Sub-agent '{sub_name_raw}' defined in frontmatter but not found in instructions")

    # Get main instruction (before sub_agent sections)
    instruction = _get_main_instruction(skill.instructions)

    # Build the agent
    agent_kwargs = {
        "name": name,
        "model": model,
        "description": fm.description,
        "instruction": instruction,
    }

    if tools and not sub_agents:
        agent_kwargs["tools"] = tools
    if sub_agents:
        agent_kwargs["sub_agents"] = sub_agents

    return Agent(**agent_kwargs)
