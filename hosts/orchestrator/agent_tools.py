"""BeeAI Tools wrapping RemoteAgentConnection."""

from typing import Any
from pydantic import BaseModel, Field
from beeai_framework.tools import Tool, StringToolOutput, ToolRunOptions
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.context import RunContext
from hosts.orchestrator.remote_agent_connection import RemoteAgentConnection
import logging

logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    query: str = Field(description="The full message, query, or instruction to send to the agent.")

class RemoteAgentTool(Tool[AgentInput, ToolRunOptions, StringToolOutput]):
    """Tool that forwards natural language queries to a remote A2A agent."""

    def __init__(self, name: str, description: str, conn: RemoteAgentConnection):
        super().__init__()
        self._name = name
        self._description = description
        self.conn = conn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self):
        return AgentInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "remote", self.name],
            creator=self,
        )

    async def _run(self, input: AgentInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """Run the tool asynchronously."""
        query = input.query
        logger.info(f"🛠️ Tool '{self.name}' invoked with: {query[:50]}...")
        
        try:
            response = await self.conn.send_message(query)
            text = response.get("text", "")
            if not text:
                return StringToolOutput("El agente no devolvió ninguna respuesta de texto.")
            return StringToolOutput(text)
        except Exception as e:
            logger.error(f"❌ Tool '{self.name}' error: {e}")
            return StringToolOutput(f"Error comunicándose con el agente: {str(e)}")
