from typing import Dict, Any, Protocol

class AgentCapabilities(Protocol):
    def get_agent_card(self) -> Dict[str, Any]:
        """Returns the agent's metadata card"""
        ...
    
    def process_request(self, message: str, context: Dict[str, Any] = None) -> str:
        """Processes a direct text request and returns a text response"""
        ...

class BaseAgentAdapter:
    """
    Base class to adapt any external agent (LangChain, AutoGen, etc.) 
    to the RubricAI A2A architecture.
    """
    def get_agent_card(self) -> Dict[str, Any]:
        raise NotImplementedError
        
    def process_request(self, message: str, context: Dict[str, Any] = None) -> str:
        raise NotImplementedError
