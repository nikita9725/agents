"""Публичная модель задачи и асинхронный сервис, независимые от исполнителя."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Literal, Protocol
from uuid import UUID

from pydantic import Field

from etl_agent.domain import AgentState, HistoryEntry, Model, RunInput

TaskStatus = Literal["pending", "running", "success", "error", "timeout", "interrupted"]
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


class TaskProgress(HistoryEntry):
    task_id: UUID
    sequence: int = Field(gt=0)


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
    async def history(self, task_id: UUID) -> list[HistoryEntry]: ...
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
        cursor = 0
        previous: TaskStatus | None = None
        try:
            while True:
                history = await self._executor.history(task.task_id)
                # A terminal snapshot may have arrived after the previous checkpoint read.
                if task.result is not None:
                    history = task.result.history
                for index in range(cursor, len(history)):
                    yield TaskEvent(
                        event="progress",
                        data=TaskProgress(
                            task_id=task.task_id,
                            sequence=index + 1,
                            **history[index].model_dump(),
                        ),
                    )
                cursor = len(history)
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
