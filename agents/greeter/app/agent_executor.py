"""Agent executor for the Greeter agent (A2A protocol)."""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Part,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import completed_task, new_artifact
from a2a.utils.errors import ServerError
from app.agent import GreetingAgent

logger = logging.getLogger(__name__)


class GreeterAgentExecutor(AgentExecutor):
    """A2A executor for the Greeter agent."""

    def __init__(self, api_key: str) -> None:
        self.agent = GreetingAgent(api_key=api_key)

    async def cancel(self, context: RequestContext) -> None:
        """Cancel is a no-op for this simple agent."""
        logger.info(f"🛑 Cancel requested for task: {context.task_id}")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the greeting request.

        Args:
            context: Request context containing user input
            event_queue: Queue for sending events back to client
        """
        query = context.get_user_input()
        if not query or not query.strip():
            raise ServerError(error=UnsupportedOperationError())

        logger.info(f"📨 Greeter received: {query[:100]}")

        try:
            result = self.agent.invoke(query, context.context_id)

            parts: list[Part] = [Part(root=TextPart(text=result))]

            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f"greeting_{context.task_id}")],
                    [context.message],
                )
            )
            logger.info("✅ Greeter response sent")

        except Exception as e:
            error_msg = f"Greeter error: {str(e)}"
            logger.exception(error_msg)

            parts = [Part(root=TextPart(text=f"Lo siento, hubo un error: {str(e)}"))]
            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f"error_{context.task_id}")],
                    [context.message],
                )
            )
