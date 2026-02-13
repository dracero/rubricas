"""BeeAI Router Implementation using ReAct Agent."""

import os
import logging
from typing import List, Optional

from beeai_framework.agents.react import ReActAgent
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.memory import UnconstrainedMemory
from hosts.orchestrator.agent_tools import RemoteAgentTool
from beeai_framework.agents.react.runners.default.runner import DefaultRunner
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode, LinePrefixParserOptions
from beeai_framework.parsers.field import ParserField
from beeai_framework.utils.strings import create_strenum
from beeai_framework.agents.react.runners.base import BaseRunner
from functools import cached_property
from typing import Callable, Any
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class GeminiRunner(DefaultRunner):
    def _create_parser(self) -> LinePrefixParser:
        tool_names = create_strenum("ToolsEnum", [tool.name for tool in self._input.tools])

        return LinePrefixParser(
            nodes={
                "thought": LinePrefixParserNode(
                    prefix="Thought: ",
                    field=ParserField.from_type(str),
                    is_start=True,
                    next=["tool_name", "final_answer"],
                ),
                "tool_name": LinePrefixParserNode(
                    prefix="Function Name: ",
                    field=ParserField.from_type(tool_names, trim=True),
                    is_start=True, # Allow starting with Function Name
                    next=["tool_input"],
                ),
                "tool_input": LinePrefixParserNode(
                    prefix="Function Input: ",
                    field=ParserField.from_type(dict, trim=True),
                    next=["tool_output"],
                    is_end=True,
                ),
                "tool_output": LinePrefixParserNode(
                    prefix="Function Output: ", field=ParserField.from_type(str), is_end=True, next=["final_answer"]
                ),
                "final_answer": LinePrefixParserNode(
                    prefix="Final Answer: ", field=ParserField.from_type(str), is_end=True, is_start=True
                ),
            },
            options=LinePrefixParserOptions(
                wait_for_start_node=True, # Still wait for a valid start node (Thought, Function Name, or Final Answer)
                end_on_repeat=True,
                fallback=lambda value: [
                    {"key": "thought", "value": "I now know the final answer."},
                    {"key": "final_answer", "value": value},
                ]
                if value
                else [],
            ),
        )

class GeminiReActAgent(ReActAgent):
    @cached_property
    def _runner(self) -> Callable[..., BaseRunner]:
        return GeminiRunner

class BeeRouter:
    """Orchestrator router using BeeAI Framework."""
    
    def __init__(self, tools: List[RemoteAgentTool], model_name: str = "gemini-2.5-flash"):
        """Initialize the router with tools and LLM."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not found in environment variables")
        
        self.llm = GeminiChatModel(
            model_id=model_name,
            api_key=api_key,
        )
        self.memory = UnconstrainedMemory()
        
        self.agent = GeminiReActAgent(
            llm=self.llm,
            tools=tools,
            memory=self.memory,
        )

    @traceable(name="BeeRouter.route", run_type="chain")
    async def route(self, user_message: str) -> str:
        """Route the user message to the appropriate agent via BeeAI."""
        try:
            # Initialize system prompt if memory is empty
            if not self.memory.messages:
                from beeai_framework.backend.message import SystemMessage
                await self.memory.add(SystemMessage(content="""Eres el Orquestador del Sistema de Rúbricas.
            Tu objetivo es usar las herramientas disponibles para responder al usuario.
            
            HERRAMIENTAS:
            - 'greeting_tool': Para saludos.
            - 'generator_tool': Para generar rúbricas.
            - 'evaluator_tool': Para evaluar documentos.

            INSTRUCCIONES CLAVE DE FORMATO:
            Tu respuesta DEBE seguir estrictamente el formato ReAct.
            1. SIEMPRE empieza con "Thought: " explicando tu razonamiento en una línea.
            2. Inmediatamente después, si necesitas usar una herramienta, pon "Function Name: " y "Function Input: ".
            
            EJEMPLO CORRECTO:
            Thought: El usuario está saludando, debo usar la herramienta de saludos.
            Function Name: greeting_tool
            Function Input: {"query": "Hola"}
            
            NO omitas nunca la línea "Thought: ".
            
            IMPORTANTE SOBRE GENERATOR Y EVALUATOR:
            - Si decidiste usar 'generator_tool', tu respuesta FINAL DEBE EMPEZAR con "ACTION:GENERATOR ".
              INCLUSO si la herramienta te devuelve una pregunta o pide más datos, tu respuesta al usuario DEBE empezar con "ACTION:GENERATOR " seguido del texto de la herramienta.
              Ejemplo: "ACTION:GENERATOR ¿Para qué nivel educativo necesitas la rúbrica?"
            
            - Si decidiste usar 'evaluator_tool', tu respuesta FINAL DEBE EMPEZAR con "ACTION:EVALUATOR ".
              INCLUSO si la herramienta pide el documento, tu respuesta al usuario DEBE empezar con "ACTION:EVALUATOR " seguido del texto.
              Ejemplo: "ACTION:EVALUATOR Por favor sube el documento que deseas evaluar."
            """))

            logger.info(f"🐝 BeeRouter processing: {user_message[:50]}...")
            response = await self.agent.run(user_message)
            
            # Extract text from ReActAgentResponse
            
            # Extract text from ReActAgentResponse
            if hasattr(response, 'result') and hasattr(response.result, 'text'):
                return response.result.text
            elif hasattr(response, 'last_message') and response.last_message:
                content = response.last_message.content
                if isinstance(content, list):
                    return "".join([c.text for c in content if hasattr(c, "text")])
                return str(content)
            
            return str(response)

        except Exception as e:
            # Check for ChatModelError and unwrap it
            error_msg = str(e)
            cause = getattr(e, "__cause__", None) or e
            cause_msg = str(cause)
            
            if "429" in error_msg or "Quota exceeded" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or \
               "429" in cause_msg or "Quota exceeded" in cause_msg or "RESOURCE_EXHAUSTED" in cause_msg:
                logger.warning(f"⚠️ Gemini Rate Limit: {cause_msg}")
                return "El sistema está recibiendo muchas solicitudes en este momento (Límite de cuota Gemini). Por favor, intenta de nuevo en un minuto."
            
            logger.error(f"❌ BeeRouter error: {e}")
            if cause:
                logger.error(f"   Caused by: {cause}")
                
            return f"Hubo un error en el orquestador inteligente: {error_msg}"
