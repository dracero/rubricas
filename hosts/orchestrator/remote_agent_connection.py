"""Remote Agent Connection - Connects to A2A agent servers.

Discovers agent capabilities via /.well-known/agent.json and
sends requests using the A2A protocol.
"""

import logging
import httpx
import json
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RemoteAgentConnection:
    """Manages connection to a remote A2A agent."""

    def __init__(self, agent_url: str):
        """Initialize connection to a remote agent.

        Args:
            agent_url: Base URL of the agent server (e.g., http://localhost:10001)
        """
        self.agent_url = agent_url.rstrip("/")
        self.agent_card: Optional[Dict[str, Any]] = None
        self._client = httpx.AsyncClient(timeout=600.0)  # 10min for full pipeline

    async def discover(self) -> Dict[str, Any]:
        """Discover agent capabilities via /.well-known/agent.json.

        Returns:
            Agent card dictionary with capabilities and skills
        """
        url = f"{self.agent_url}/.well-known/agent.json"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            self.agent_card = response.json()
            logger.info(
                f"✅ Discovered agent: {self.agent_card.get('name', 'unknown')} "
                f"at {self.agent_url}"
            )
            return self.agent_card
        except Exception as e:
            logger.error(f"❌ Failed to discover agent at {url}: {e}")
            raise

    def _get_trace_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject LangSmith distributed tracing headers."""
        try:
            from langsmith.run_helpers import get_current_run_tree
            rt = get_current_run_tree()
            if rt:
                trace_headers = rt.to_headers()
                if trace_headers:
                    headers.update(trace_headers)
                    logger.debug(f"🔗 Injected trace headers: {trace_headers.keys()}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ Failed to inject trace headers: {e}")
        return headers

    async def discover(self) -> Dict[str, Any]:
        """Discover agent capabilities via /.well-known/agent.json.

        Returns:
            Agent card dictionary with capabilities and skills
        """
        url = f"{self.agent_url}/.well-known/agent.json"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            self.agent_card = response.json()
            logger.info(
                f"✅ Discovered agent: {self.agent_card.get('name', 'unknown')} "
                f"at {self.agent_url}"
            )
            return self.agent_card
        except Exception as e:
            logger.error(f"❌ Failed to discover agent at {url}: {e}")
            raise

    async def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the agent and get the response.

        Uses the A2A JSON-RPC protocol to send messages.

        Args:
            message: User message to send

        Returns:
            Response dictionary with the agent's reply
        """
        message_id = str(uuid.uuid4())
        context_id = str(uuid.uuid4())

        # A2A JSON-RPC request (message/send method)
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                    "messageId": message_id,
                    "contextId": context_id,
                },
            },
        }

        try:
            response = await self._client.post(
                self.agent_url,
                json=payload,
                headers=self._get_trace_headers({"Content-Type": "application/json"}),
            )
            response.raise_for_status()
            result = response.json()

            # Extract text from A2A response
            text = self._extract_text(result)
            return {
                "task_id": message_id,
                "text": text,
                "raw": result,
            }
        except Exception as e:
            logger.error(f"❌ Error sending message to {self.agent_url}: {e}")
            raise

    def _extract_text(self, response: Dict) -> str:
        """Extract text content from an A2A JSON-RPC response."""
        try:
            result = response.get("result", {})
            logger.debug(f"🔍 Raw A2A result: {json.dumps(result, default=str)[:500]}")

            # Helper to extract text from parts list
            def texts_from_parts(parts):
                texts = []
                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            texts.append(part["text"])
                        root = part.get("root", {})
                        if isinstance(root, dict) and "text" in root:
                            texts.append(root["text"])
                return texts

            # Path 1: result → artifacts → parts → text (ALL artifacts)
            artifacts = result.get("artifacts", [])
            if artifacts:
                all_texts = []
                for artifact in artifacts:
                    parts = artifact.get("parts", [])
                    texts = texts_from_parts(parts)
                    all_texts.extend(texts)
                if all_texts:
                    return "\n".join(all_texts)

            # Path 2: result → status → message → parts → text
            status = result.get("status", {})
            if isinstance(status, dict):
                msg = status.get("message", {})
                if isinstance(msg, dict) and "parts" in msg:
                    texts = texts_from_parts(msg["parts"])
                    if texts:
                        return "\n".join(texts)

            # Path 3: result is the full Task; check result → result → ...
            inner_result = result.get("result", {})
            if isinstance(inner_result, dict):
                # Task result with status & artifacts
                inner_artifacts = inner_result.get("artifacts", [])
                if inner_artifacts:
                    parts = inner_artifacts[0].get("parts", [])
                    texts = texts_from_parts(parts)
                    if texts:
                        return "\n".join(texts)

            # Path 4: check for error in JSON-RPC response
            error = response.get("error", {})
            if error:
                return f"Error from agent: {error.get('message', str(error))}"

            logger.warning(f"⚠️ Could not find text in response. Keys: {list(result.keys())}")
            return str(result)
        except Exception as e:
            logger.warning(f"Could not extract text: {e}")
            return str(response)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def get_description(self) -> str:
        """Get agent description for routing decisions."""
        if self.agent_card:
            return self.agent_card.get("description", "No description")
        return "Undiscovered agent"

    def get_name(self) -> str:
        """Get agent name."""
        if self.agent_card:
            return self.agent_card.get("name", "Unknown")
        return "Unknown"
