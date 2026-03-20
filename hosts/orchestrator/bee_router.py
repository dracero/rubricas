"""Simple LLM Router using Groq for classification."""

import os
import logging
from typing import List, Optional

from hosts.orchestrator.agent_tools import RemoteAgentTool

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimpleRouter:
    """Simple orchestrator router using Groq LLM for classification."""
    
    def __init__(self, tools: List[RemoteAgentTool], model_name: Optional[str] = None):
        """Initialize the router with tools and LLM."""
        self.tools = {tool.name: tool for tool in tools}
        self.model_name = model_name or os.environ.get("ORCHESTRATOR_MODEL", "llama-3.3-70b-versatile")
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("⚠️ GROQ_API_KEY not found, falling back to keyword matching")
            self.client = None
        elif not GROQ_AVAILABLE:
            logger.warning("⚠️ groq package not installed, falling back to keyword matching")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
        
        logger.info(f"✅ SimpleRouter initialized with tools: {list(self.tools.keys())}")

    @traceable(name="SimpleRouter.route", run_type="chain")
    async def route(self, user_message: str) -> str:
        """Route the user message to the appropriate agent via LLM classification."""
        try:
            # Use LLM to classify the intent
            if self.client:
                classification = await self._classify_with_llm(user_message)
            else:
                classification = self._classify_with_keywords(user_message)
            
            logger.info(f"🎯 Classification: {classification}")
            
            # Route based on classification
            if classification == "evaluator":
                tool = self.tools.get("evaluator_tool")
                if tool:
                    result = await tool.run(user_message)
                    return f"ACTION:EVALUATOR {result}"
                return "Error: evaluator_tool no disponible"
            
            elif classification == "generator":
                tool = self.tools.get("generator_tool")
                if tool:
                    result = await tool.run(user_message)
                    return f"ACTION:GENERATOR {result}"
                return "Error: generator_tool no disponible"
            
            else:  # greeter
                tool = self.tools.get("greeter_tool")
                if tool:
                    result = await tool.run(user_message)
                    return result  # Sin prefijo ACTION:GREETER
                return "Error: greeter_tool no disponible"

        except Exception as e:
            logger.exception(f"❌ SimpleRouter error: {e}")
            return f"Hubo un error en el orquestador: {str(e)}"

    async def _classify_with_llm(self, user_message: str) -> str:
        """Classify user intent using Groq LLM."""
        prompt = f"""Clasifica la siguiente solicitud del usuario en UNA de estas categorías:

- generator: Si el usuario quiere CREAR, GENERAR o DISEÑAR una rúbrica
- evaluator: Si el usuario quiere EVALUAR, CALIFICAR o REVISAR un documento
- greeter: Si es un saludo, pregunta general, o no está claro

Solicitud del usuario: "{user_message}"

Responde SOLO con una palabra: generator, evaluator, o greeter"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Validate response
            if classification in ["generator", "evaluator", "greeter"]:
                return classification
            
            # Fallback if LLM returns something unexpected
            logger.warning(f"⚠️ Unexpected LLM response: {classification}, using keyword fallback")
            return self._classify_with_keywords(user_message)
            
        except Exception as e:
            logger.error(f"❌ LLM classification error: {e}, using keyword fallback")
            return self._classify_with_keywords(user_message)

    def _classify_with_keywords(self, user_message: str) -> str:
        """Fallback classification using keyword matching."""
        import re
        message_lower = user_message.lower()
        
        # Keywords for each category
        generator_keywords = [
            r'\bgenerar\b', r'\bcrear\b', r'\bdiseñar\b', r'\brúbrica\b',
            r'\brubrica\b', r'\bnueva\b', r'\bhacer\b', r'\bconstruir\b'
        ]
        
        evaluator_keywords = [
            r'\bevaluar\b', r'\bcalificar\b', r'\brevisar\b', r'\banalizar\b',
            r'\bcorregir\b', r'\bpuntuar\b'
        ]
        
        greeter_keywords = [
            r'\bhola\b', r'\bhi\b', r'\bhello\b', r'\bbuenas\b', 
            r'\bqué tal\b', r'\bayuda\b', r'\bhelp\b'
        ]
        
        # Check for matches
        generator_match = any(re.search(pattern, message_lower) for pattern in generator_keywords)
        evaluator_match = any(re.search(pattern, message_lower) for pattern in evaluator_keywords)
        greeter_match = any(re.search(pattern, message_lower) for pattern in greeter_keywords)
        
        # Priority: evaluator > generator > greeter
        if evaluator_match and not generator_match:
            return "evaluator"
        elif generator_match:
            return "generator"
        else:
            return "greeter"
