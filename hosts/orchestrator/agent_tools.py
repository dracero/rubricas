"""Simple wrapper for RemoteAgentConnection without BeeAI dependency."""

from hosts.orchestrator.remote_agent_connection import RemoteAgentConnection
import logging

logger = logging.getLogger(__name__)


class RemoteAgentTool:
    """Simple tool that forwards queries to a remote A2A agent."""

    def __init__(self, name: str, description: str, conn: RemoteAgentConnection):
        self.name = name
        self.description = description
        self.conn = conn

    async def run(self, query: str) -> str:
        """Run the tool by sending a message to the remote agent."""
        logger.info(f"🛠️ Tool '{self.name}' invoked with: {query[:50]}...")
        
        try:
            response = await self.conn.send_message(query)
            text = response.get("text", "")
            logger.info(f"📥 Tool '{self.name}' received response ({len(text)} chars)")
            if not text:
                return "El agente no devolvió ninguna respuesta de texto."
            return text
        except Exception as e:
            logger.error(f"❌ Tool '{self.name}' error: {e}")
            return f"Error comunicándose con el agente: {str(e)}"
