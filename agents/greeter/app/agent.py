"""Greeter Agent - Handles greetings and system explanations.

Uses LangGraph + Google Gemini to provide friendly responses
about the RubricAI system.
"""

import logging
from typing import Any, Dict, List, TypedDict, Annotated
import operator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END, START
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = ["text", "text/*"]


class AgentState(TypedDict):
    """The state of the greeting agent."""
    messages: Annotated[List[BaseMessage], operator.add]


def generate_greeting(state: AgentState, model: ChatGroq):
    """Node that generates a greeting response."""
    messages = state['messages']
    last_message = messages[-1].content if messages else ""

    prompt_template = ChatPromptTemplate.from_template(
        "Eres un asistente AMIGABLE y ENTUSIASTA del sistema RubricAI. "
        "Tu único trabajo es SALUDAR a los usuarios, explicarles qué hace el sistema "
        "(generar y evaluar rúbricas académicas usando IA) y preguntarles qué necesitan. "
        "Usa emojis. Sé breve pero cálido. \n\n"
        "Usuario: {input}"
    )
    
    chain = prompt_template | model
    try:
        response = chain.invoke({"input": last_message})
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        return {"messages": [AIMessage(content="Lo siento, tuve un problema al generar el saludo.")]}


class GreetingAgent:
    """Agent that greets users and explains the RubricAI system using LangGraph."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/*"]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.app = None

    def _ensure_initialized(self):
        """Lazy-initialize the LangGraph workflow."""
        if self.app:
            return

        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "Missing dependencies for Greeter agent. "
                "Please ensure 'langgraph' is installed."
            )

        logger.info("🔧 Initializing Greeter LangGraph...")

        # Initialize LLM
        model = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=self.api_key,
            temperature=0.7,
        )

        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        # Use partial to inject dependencies if needed, or simple lambda
        workflow.add_node("greeter", lambda state: generate_greeting(state, model))
        
        # Add edges
        workflow.add_edge(START, "greeter")
        workflow.add_edge("greeter", END)

        self.app = workflow.compile()
        logger.info("✅ Greeter LangGraph ready")

    def invoke(self, query: str, session_id: str = None) -> str:
        """Process a greeting using the graph."""
        self._ensure_initialized()

        logger.info(f"🤖 Greeter processing: {query[:80]}...")
        
        # Guardrail: Truncate query to 2,000 chars for Greeter
        truncated_query = query[:2000]
        inputs = {"messages": [HumanMessage(content=truncated_query)]}
        result = self.app.invoke(inputs)
        
        # Extract the last message content
        last_message = result['messages'][-1]
        return last_message.content
