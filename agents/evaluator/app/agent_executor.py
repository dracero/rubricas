"""Agent executor for the Rubric Evaluator agent (A2A protocol)."""

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
from dotenv import load_dotenv
load_dotenv() # Load env vars before importing agent

from app.agent import RubricEvaluatorAgent

logger = logging.getLogger(__name__)


class RubricEvaluatorAgentExecutor(AgentExecutor):
    """A2A executor for the Rubric Evaluator agent."""

    def __init__(self) -> None:
        self.agent = RubricEvaluatorAgent()
        self._is_cancelled = False

    async def cancel(self, context: RequestContext) -> None:
        """Cancel the current execution."""
        logger.info(f"🛑 Cancel requested for task: {context.task_id}")
        self._is_cancelled = True

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the evaluation request.

        Args:
            context: Request context containing user input
            event_queue: Queue for sending events back to client
        """
        self._is_cancelled = False
        query = context.get_user_input()

        if not query or not query.strip():
            raise ServerError(error=UnsupportedOperationError())

        logger.info(f"📨 Evaluator received: {query[:100]}")

        try:
            if self._is_cancelled:
                return

            result = self.agent.invoke(query, context.context_id)

            if self._is_cancelled:
                return

            if not result:
                result = "No se pudo realizar la evaluación. Intente de nuevo."

            parts: list[Part] = [Part(root=TextPart(text=str(result)))]

            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f"evaluation_{context.task_id}")],
                    [context.message],
                )
            )
            logger.info("✅ Evaluator response sent")

        except Exception as e:
            error_msg = f"Evaluator error: {str(e)}"
            logger.exception(error_msg)

            parts = [
                Part(root=TextPart(text=f"Error al evaluar: {str(e)}"))
            ]
            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f"error_{context.task_id}")],
                    [context.message],
                )
            )
