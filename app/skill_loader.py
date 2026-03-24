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

import yaml
from google.adk.agents import Agent

logger = logging.getLogger(__name__)


def _parse_skill_file(filepath: Path) -> Dict[str, Any]:
    """Parse a .md skill file into frontmatter dict + body string.

    Returns:
        Dict with keys: 'meta' (dict from YAML frontmatter) and 'body' (str).
    """
    content = filepath.read_text(encoding="utf-8")

    # Extract YAML frontmatter between --- delimiters
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
    if not fm_match:
        raise ValueError(f"Skill file {filepath} missing YAML frontmatter (---)")

    meta = yaml.safe_load(fm_match.group(1)) or {}
    body = fm_match.group(2).strip()

    return {"meta": meta, "body": body}


def _parse_sub_agents_from_body(body: str) -> Dict[str, Dict[str, Any]]:
    """Parse sub-agent definitions from the skill body.

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
    # Then alternates: name, content, name, content...
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
    default_model: str = "gemini-2.5-flash",
) -> List[Agent]:
    """Load all skill files from a directory and create ADK Agents.

    Args:
        skills_dir: Path to directory containing .md skill files.
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
    skill_files = sorted(skills_path.glob("*.md"))

    if not skill_files:
        logger.info(f"📂 No .md skill files found in {skills_dir} (upload skills via frontend)")
        return []

    logger.info(f"📂 Loading {len(skill_files)} skills from {skills_dir}")

    for filepath in skill_files:
        try:
            agent = _load_single_skill(filepath, available_tools, default_model)
            agents.append(agent)
            logger.info(f"  ✅ Loaded skill: {agent.name} ({filepath.name})")
        except Exception as e:
            logger.error(f"  ❌ Error loading skill {filepath.name}: {e}")

    logger.info(f"🎯 Loaded {len(agents)} skills total")
    return agents


def _load_single_skill(
    filepath: Path,
    available_tools: Dict[str, Callable],
    default_model: str,
) -> Agent:
    """Load a single skill file and create an ADK Agent."""
    parsed = _parse_skill_file(filepath)
    meta = parsed["meta"]
    body = parsed["body"]

    name = meta.get("name", filepath.stem)
    model = meta.get("model", default_model)
    tool_names = meta.get("tools", [])
    sub_agent_names = meta.get("sub_agents", [])

    # Resolve tool functions
    tools = []
    for tool_name in tool_names:
        if tool_name in available_tools:
            tools.append(available_tools[tool_name])
        else:
            logger.warning(f"  ⚠️ Tool '{tool_name}' not found for skill '{name}'")

    # Parse sub-agents from body
    sub_agents_defs = _parse_sub_agents_from_body(body)

    # Create sub-agent instances
    sub_agents = []
    for sub_name in sub_agent_names:
        if sub_name in sub_agents_defs:
            sub_def = sub_agents_defs[sub_name]

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
            logger.warning(f"  ⚠️ Sub-agent '{sub_name}' defined in frontmatter but not found in body")

    # Get main instruction (before sub_agent sections)
    instruction = _get_main_instruction(body)

    # Build the agent
    agent_kwargs = {
        "name": name,
        "model": model,
        "instruction": instruction,
    }

    if tools and not sub_agents:
        agent_kwargs["tools"] = tools
    if sub_agents:
        agent_kwargs["sub_agents"] = sub_agents

    return Agent(**agent_kwargs)


def load_skill_from_file(
    filepath: str,
    available_tools: Dict[str, Callable],
    default_model: str = "gemini-2.5-flash",
) -> Agent:
    """Load a single skill from a specific file path.

    This function allows loading skills from any location,
    not just the skills directory.

    Args:
        filepath: Absolute path to the .md skill file.
        available_tools: Dict mapping tool function names to actual callables.
        default_model: Default model to use if not specified in skill file.

    Returns:
        An ADK Agent instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {filepath}")
    if not path.suffix == ".md":
        raise ValueError(f"Skill file must be .md: {filepath}")

    return _load_single_skill(path, available_tools, default_model)
