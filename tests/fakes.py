"""Тестовые адаптеры, недоступные для выбора через публичный HTTP API."""

from dataclasses import dataclass, field

from dbos import EnqueueOptions
from langchain_core.messages import AIMessage, AnyMessage

from etl_agent.application import Planner
from etl_agent.domain import FinalResponse, NextAction
from etl_agent.exceptions import PlanningError
from etl_agent.infrastructure.task_executor import WorkflowRecord


class FakePlanner(Planner):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def propose(self, goal: str) -> NextAction:
        self.calls.append(goal)
        if "permission" in goal.lower():
            action = "Проверить права учётной записи загрузки на чтение источника."
        elif "формат" in goal.lower():
            action = "Сверить формат проблемного поля CSV со схемой загрузки."
        else:
            action = "Проверить наличие входного файла для указанного запуска ETL."
        return NextAction(action=action, rationale=f"Сначала нужно проверить сведения: {goal}")

    async def plan(self, messages: list[AnyMessage]) -> AIMessage:
        action = await self.propose(str(messages[0].content))
        return AIMessage(content=f"{action.action}\n{action.rationale}")

    async def finalize(self, messages: list[AnyMessage]) -> FinalResponse:
        return FinalResponse(
            final_answer=str(messages[-1].content),
            next_action=NextAction(
                action="Проверить входные данные", rationale="На основании наблюдений"
            ),
        )


class FailingPlanner(Planner):
    @staticmethod
    async def propose(goal: str) -> NextAction:
        raise PlanningError("private provider details must not escape")


@dataclass(slots=True)
class FakeWorkflow:
    status: str
    workflow_deadline_epoch_ms: int | None = None
    completed_at: int | None = None
    output: object = None


@dataclass(slots=True)
class EnqueuedTask:
    options: EnqueueOptions
    task_id: str
    goal: str
    max_steps: int
    correlation: str | None


@dataclass(slots=True)
class FakeWorkflowClient:
    workflows: list[WorkflowRecord] = field(default_factory=list)
    enqueued: list[EnqueuedTask] = field(default_factory=list)
    enqueue_error: Exception | None = None
    enqueue_attempts: int = 0
    ready_calls: int = 0
    destroy_calls: int = 0
    cancelled: list[str] = field(default_factory=list)

    async def enqueue_async(
        self,
        options: EnqueueOptions,
        task_id: str,
        goal: str,
        max_steps: int,
        correlation: str | None,
    ) -> None:
        self.enqueue_attempts += 1
        if self.enqueue_error is not None:
            raise self.enqueue_error
        self.enqueued.append(EnqueuedTask(options, task_id, goal, max_steps, correlation))

    async def list_workflows_async(
        self,
        *,
        workflow_ids: list[str],
        name: str,
        load_input: bool,
    ) -> list[WorkflowRecord]:
        return self.workflows

    async def cancel_workflow_async(self, workflow_id: str) -> None:
        self.cancelled.append(workflow_id)

    async def check_connection_async(self) -> None:
        self.ready_calls += 1

    def destroy(self) -> None:
        self.destroy_calls += 1
