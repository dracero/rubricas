from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum

class AgentType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    USER = "user"

class MessageType(str, Enum):
    TEXT = "text"
    ACTION_REQUEST = "action_request"
    ACTION_RESPONSE = "action_response"
    ERROR = "error"

class AgentMessage(BaseModel):
    source: str
    target: str
    type: MessageType
    content: str
    metadata: Dict[str, Any] = {}

class ConversationState(BaseModel):
    conversation_id: str
    history: List[AgentMessage] = []
    context: Dict[str, Any] = {}
