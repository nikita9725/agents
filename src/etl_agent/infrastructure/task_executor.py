"""DBOS хранит очередь и статусы, LangGraph — checkpoints в PostgreSQL."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from dbos import DBOSClient, EnqueueOptions, WorkflowSerializationFormat, WorkflowStatus
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from etl_agent.domain import AgentState, HistoryEntry, RunInput
from etl_agent.observability import request_id
from etl_agent.runtime import QUEUE_NAME, WORKFLOW_NAME, RuntimeSettings, checkpoints
from etl_agent.tasks import ExecutorUnavailable, NodeName, Task, TaskError, TaskNotFound, TaskStatus

logger = logging.getLogger(__name__)
STATUS_MAP: dict[str, TaskStatus] = {
    "DELAYED": "pending",
    "ENQUEUED": "pending",
    "PENDING": "running",
    "SUCCESS": "success",
    "ERROR": "error",
    "MAX_RECOVERY_ATTEMPTS_EXCEEDED": "error",
    "CANCELLED": "interrupted",
}


class Checkpoint(BaseModel):
    history: list[HistoryEntry] = Field(default_factory=list)


class DBOSTasks:
    def __init__(self, client: DBOSClient, settings: RuntimeSettings) -> None:
        self._client = client
        self._settings = settings

    @asynccontextmanager
    async def _errors(self) -> AsyncIterator[None]:
        try:
            yield
        except TaskNotFound:
            raise
        except Exception as exc:
            logger.exception("event=executor.failed")
            raise ExecutorUnavailable from exc

    async def _workflow(self, task_id: UUID) -> WorkflowStatus:
        workflows = await self._client.list_workflows_async(
            workflow_ids=[str(task_id)],
            name=WORKFLOW_NAME,
            load_input=False,
        )
        if not workflows:
            raise TaskNotFound
        return workflows[0]

    async def create(self, request: RunInput) -> Task:
        async with self._errors():
            task_id = uuid4()
            options: EnqueueOptions = {
                "workflow_name": WORKFLOW_NAME,
                "queue_name": QUEUE_NAME,
                "workflow_id": str(task_id),
                "workflow_timeout": 180,
                "serialization_type": WorkflowSerializationFormat.PORTABLE,
            }
            await self._client.enqueue_async(
                options, str(task_id), request.goal, request.max_steps, request_id.get()
            )
            logger.info("event=task.created task_id=%s", task_id)
            return Task(task_id=task_id, status="pending")

    async def get(self, task_id: UUID) -> Task:
        async with self._errors():
            workflow = await self._workflow(task_id)
            status = STATUS_MAP[workflow.status]
            if (
                status == "interrupted"
                and workflow.workflow_deadline_epoch_ms is not None
                and workflow.completed_at is not None
                and workflow.completed_at >= workflow.workflow_deadline_epoch_ms
            ):
                status = "timeout"
            result = None
            error = None
            if status == "success":
                output: object = workflow.output
                if not isinstance(output, str):
                    raise ValueError("Invalid task output")
                result = AgentState.model_validate_json(output)
            elif status in {"error", "timeout"}:
                error = TaskError(
                    code="timeout" if status == "timeout" else "execution_failed",
                    message="Task execution timed out."
                    if status == "timeout"
                    else "Task execution failed. See server logs for diagnostics.",
                )
            return Task(task_id=task_id, status=status, result=result, error=error)

    async def cancel(self, task_id: UUID) -> Task:
        async with self._errors():
            workflow = await self._workflow(task_id)
            if workflow.status in {"DELAYED", "ENQUEUED", "PENDING"}:
                await self._client.cancel_workflow_async(str(task_id))
            return await self.get(task_id)

    async def completed_nodes(self, task_id: UUID) -> list[NodeName]:
        async with self._errors():
            config: RunnableConfig = {"configurable": {"thread_id": str(task_id)}}
            async with checkpoints(self._settings) as saver:
                checkpoint = await saver.aget_tuple(config)
                if checkpoint is None:
                    return []
                state = Checkpoint.model_validate(checkpoint.checkpoint["channel_values"])
                return [entry.node for entry in state.history]

    async def ready(self) -> None:
        async with self._errors():
            await self._client.check_connection_async()
