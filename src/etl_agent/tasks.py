"""Публичная модель задачи и асинхронный сервис, независимые от исполнителя."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Literal, Protocol
from uuid import UUID

from etl_agent.domain import AgentState, Model, RunInput

TaskStatus = Literal["pending", "running", "success", "error", "timeout", "interrupted"]
NodeName = Literal["input", "planning", "finalization"]
TERMINAL = {"success", "error", "timeout", "interrupted"}


class TaskError(Model):
    code: Literal[
        "execution_failed",
        "timeout",
        "executor_unavailable",
        "internal_error",
        "not_found",
        "invalid_input",
    ]
    message: str


class Task(Model):
    task_id: UUID
    status: TaskStatus
    result: AgentState | None = None
    error: TaskError | None = None


class TaskProgress(Model):
    task_id: UUID
    node: NodeName


class TaskStreamError(Model):
    task_id: UUID
    error: TaskError


class TaskEvent(Model):
    event: Literal["task", "progress", "error"]
    data: Task | TaskProgress | TaskStreamError

    def encode(self) -> str:
        return f"event: {self.event}\ndata: {self.data.model_dump_json()}\n\n"


class TaskNotFound(Exception):
    pass


class ExecutorUnavailable(Exception):
    pass


class TaskExecutor(Protocol):
    async def create(self, request: RunInput) -> Task: ...
    async def get(self, task_id: UUID) -> Task: ...
    async def cancel(self, task_id: UUID) -> Task: ...
    async def completed_nodes(self, task_id: UUID) -> list[NodeName]: ...
    async def ready(self) -> None: ...


class TaskService:
    def __init__(self, executor: TaskExecutor) -> None:
        self._executor = executor

    async def create(self, request: RunInput) -> Task:
        return await self._executor.create(request)

    async def get(self, task_id: UUID) -> Task:
        return await self._executor.get(task_id)

    async def cancel(self, task_id: UUID) -> Task:
        return await self._executor.cancel(task_id)

    async def ready(self) -> None:
        await self._executor.ready()

    async def events(self, initial: Task) -> AsyncGenerator[TaskEvent]:
        # Read durable progress; disconnecting the observer never cancels execution.
        task = initial
        seen: set[NodeName] = set()
        previous: TaskStatus | None = None
        try:
            while True:
                for node in await self._executor.completed_nodes(task.task_id):
                    if node not in seen:
                        seen.add(node)
                        yield TaskEvent(
                            event="progress", data=TaskProgress(task_id=task.task_id, node=node)
                        )
                if task.status != previous:
                    yield TaskEvent(event="task", data=task)
                    previous = task.status
                if task.status in TERMINAL:
                    return
                await asyncio.sleep(0.2)
                task = await self.get(task.task_id)
        except (ExecutorUnavailable, TaskNotFound):
            yield TaskEvent(
                event="error",
                data=TaskStreamError(
                    task_id=task.task_id,
                    error=TaskError(
                        code="executor_unavailable",
                        message="Task executor is unavailable; reconnect to observe the task.",
                    ),
                ),
            )
