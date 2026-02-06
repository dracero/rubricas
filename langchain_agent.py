from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agent_adapters import BaseAgentAdapter

class GreetingAgent(BaseAgentAdapter):
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        self.prompt = ChatPromptTemplate.from_template(
            "Eres un asistente AMIGABLE y ENTUSIASTA del sistema RubricAI. "
            "Tu único trabajo es SALUDAR a los usuarios, explicarles qué hace el sistema "
            "(generar y evaluar rúbricas) y preguntarles qué necesitan. "
            "Usa emojis. Sé breve. \n\n"
            "Usuario: {input}"
        )
        self.chain = self.prompt | self.llm

    def get_agent_card(self) -> Dict[str, Any]:
        return {
            "id": "greeter",
            "name": "Agente de Bienvenida",
            "description": "Saluda a los usuarios, explica las capacidades del sistema y ofrece ayuda inicial. Usar cuando el usuario saluda (hola, buen día) o pregunta qué puede hacer el sistema.",
            "capabilities": ["saludar", "explicar sistema", "charlar"],
            "type": "social"
        }

    def process_request(self, message: str, context: Dict[str, Any] = None) -> str:
        """Procesa el mensaje usando LangChain Chain"""
        response = self.chain.invoke({"input": message})
        return response.content
