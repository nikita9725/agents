from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest
from dbos import WorkflowSerializationFormat
from starlette.requests import Request

from etl_agent.api import create_app, get_service
from etl_agent.domain import AgentState, HistoryEntry, NextAction, RunInput
from etl_agent.infrastructure.task_executor import DBOSTasks
from etl_agent.runtime import RuntimeSettings
from etl_agent.tasks import (
    ExecutorUnavailable,
    Task,
    TaskError,
    TaskNotFound,
    TaskService,
    TaskStatus,
)
from tests.fakes import FakeWorkflow, FakeWorkflowClient

pytestmark = pytest.mark.unit


class FakeExecutor:
    def __init__(self) -> None:
        self.task_id = uuid4()
        self.status: TaskStatus = "pending"
        self.unavailable = False
        self.transition = False
        self._reads = 0
        self.calls: list[str] = []

    async def ready(self) -> None:
        if self.unavailable:
            raise ExecutorUnavailable

    async def create(self, request: RunInput) -> Task:
        await self.ready()
        self.calls.append("create")
        return Task(task_id=self.task_id, status="pending")

    async def get(self, task_id: UUID) -> Task:
        await self.ready()
        self.calls.append("get")
        if task_id != self.task_id:
            raise TaskNotFound
        self._reads += 1
        if self.transition and self._reads > 1:
            self.status = "success"
        result = None
        error = None
        if self.status == "success":
            result = AgentState(
                goal="file missing",
                current_step=1,
                next_action=NextAction(action="Read logs", rationale="Need error"),
                final_answer="Read logs",
                history=[
                    HistoryEntry(
                        role="user" if node == "input" else "assistant", node=node, content="safe"
                    )
                    for node in ("input", "planning", "finalization")
                ],
            )
        elif self.status in {"error", "timeout"}:
            error = TaskError(
                code="execution_failed" if self.status == "error" else "timeout",
                message="Task failed",
            )
        return Task(task_id=task_id, status=self.status, result=result, error=error)

    async def cancel(self, task_id: UUID) -> Task:
        task = await self.get(task_id)
        if task.status in {"pending", "running"}:
            self.calls.append("cancel")
            self.status = "interrupted"
        return await self.get(task_id)

    async def history(self, task_id: UUID) -> list[HistoryEntry]:
        await self.ready()
        if self.status != "success":
            return []
        task = await self.get(task_id)
        assert task.result is not None
        return task.result.history


@asynccontextmanager
async def client_for(executor: FakeExecutor) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(TaskService(executor))
    async with httpx.AsyncClient(
        base_url="http://public", transport=httpx.ASGITransport(app=app)
    ) as client:
        yield client


async def test_create_accepts_background_work_without_waiting() -> None:
    executor = FakeExecutor()
    async with client_for(executor) as client:
        response = await client.post("/tasks", json={"goal": "file missing"})
    assert response.status_code == 202
    assert response.headers["Location"] == f"/tasks/{executor.task_id}"
    assert response.json() == {
        "task_id": str(executor.task_id),
        "status": "pending",
        "result": None,
        "error": None,
    }
    assert executor.calls == ["create"]


@pytest.mark.parametrize(
    "status", ["pending", "running", "success", "error", "timeout", "interrupted"]
)
async def test_status_result_and_error_are_public_models(status: TaskStatus) -> None:
    executor = FakeExecutor()
    executor.status = status
    async with client_for(executor) as client:
        response = await client.get(f"/tasks/{executor.task_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == status
    assert (body["result"] is not None) == (status == "success")
    assert (body["error"] is not None) == (status in {"error", "timeout"})
    assert "thread_id" not in response.text and "run_id" not in response.text


async def test_unknown_task_and_bad_uuid() -> None:
    executor = FakeExecutor()
    async with client_for(executor) as client:
        for suffix in ("", "/events", "/cancel"):
            response = await client.request(
                "POST" if suffix == "/cancel" else "GET", f"/tasks/{uuid4()}{suffix}"
            )
            assert response.status_code == 404
        assert (await client.get("/tasks/not-a-uuid")).status_code == 422


@pytest.mark.parametrize(
    "body", [{"goal": " "}, {"goal": "x", "max_steps": 0}, {"goal": "x", "thread_id": "secret"}]
)
async def test_invalid_input_never_reaches_executor(body: dict[str, str | int]) -> None:
    executor = FakeExecutor()
    async with client_for(executor) as client:
        response = await client.post("/tasks", json=body)
    assert response.status_code == 422 and "secret" not in response.text
    assert not executor.calls


@pytest.mark.parametrize(
    "status", ["pending", "running", "success", "error", "timeout", "interrupted"]
)
async def test_cancellation_is_idempotent(status: TaskStatus) -> None:
    executor = FakeExecutor()
    executor.status = status
    async with client_for(executor) as client:
        first = await client.post(f"/tasks/{executor.task_id}/cancel")
        second = await client.post(f"/tasks/{executor.task_id}/cancel")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert executor.calls.count("cancel") == (1 if status in {"pending", "running"} else 0)
    assert "create" not in executor.calls


async def test_sse_replays_progress_and_observes_completion() -> None:
    executor = FakeExecutor()
    executor.status = "running"
    executor.transition = True
    async with client_for(executor) as client:
        response = await client.get(f"/tasks/{executor.task_id}/events")
    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert response.text.count("event: progress") == 3
    assert '"status":"running"' in response.text and '"status":"success"' in response.text
    assert "thread_id" not in response.text and "run_id" not in response.text
    assert "create" not in executor.calls and "cancel" not in executor.calls


async def test_closing_observer_does_not_cancel_execution() -> None:
    executor = FakeExecutor()
    service = TaskService(executor)
    events = service.events(await service.get(executor.task_id))
    assert (await anext(events)).event == "task"
    await events.aclose()
    assert "cancel" not in executor.calls


async def test_public_schema_health_and_executor_errors() -> None:
    executor = FakeExecutor()
    async with client_for(executor) as client:
        schema = (await client.get("/openapi.json")).json()
        assert set(schema["paths"]) == {
            "/ok",
            "/ready",
            "/tasks",
            "/tasks/{task_id}",
            "/tasks/{task_id}/cancel",
            "/tasks/{task_id}/events",
        }
        for path in ("/threads", "/runs", "/assistants", "/store"):
            assert (await client.get(path)).status_code == 404
        executor.unavailable = True
        assert (await client.get("/ok")).status_code == 200
        assert (await client.get("/ready")).status_code == 503
        assert (await client.post("/tasks", json={"goal": "file missing"})).status_code == 503


async def test_applications_keep_services_isolated() -> None:
    first, second = FakeExecutor(), FakeExecutor()
    first.status, second.status = "running", "success"
    async with client_for(first) as a, client_for(second) as b:
        assert (await a.get(f"/tasks/{first.task_id}")).json()["status"] == "running"
        assert (await b.get(f"/tasks/{second.task_id}")).json()["status"] == "success"
        assert (await a.get(f"/tasks/{second.task_id}")).status_code == 404
        assert (await b.get(f"/tasks/{first.task_id}")).status_code == 404


@pytest.mark.parametrize("fail", [False, True])
async def test_lifespan_closes_owned_client_and_clears_service(fail: bool) -> None:
    client = FakeWorkflowClient()

    @asynccontextmanager
    async def runtime() -> AsyncIterator[TaskService]:
        try:
            yield TaskService(DBOSTasks(client, RuntimeSettings(_env_file=None)))
        finally:
            client.destroy()

    application = create_app(runtime_factory=runtime)
    request = Request({"type": "http", "app": application})
    with pytest.raises(ExecutorUnavailable):
        get_service(request)
    try:
        async with application.router.lifespan_context(application):
            await get_service(request).ready()
            if fail:
                raise RuntimeError("lifespan body failed")
    except RuntimeError as exc:
        assert fail and str(exc) == "lifespan body failed"
    assert client.destroy_calls == 1 and client.ready_calls == 1
    with pytest.raises(ExecutorUnavailable):
        get_service(request)


async def test_lifespan_keeps_injected_service_with_owner() -> None:
    service = TaskService(FakeExecutor())
    runtime_calls = 0

    @asynccontextmanager
    async def runtime() -> AsyncIterator[TaskService]:
        nonlocal runtime_calls
        runtime_calls += 1
        yield service

    application = create_app(service, runtime_factory=runtime)
    request = Request({"type": "http", "app": application})
    async with application.router.lifespan_context(application):
        assert get_service(request) is service
    assert runtime_calls == 0
    assert get_service(request) is service


async def test_dbos_enqueue_is_durable_and_has_same_task_id() -> None:
    client = FakeWorkflowClient()
    executor = DBOSTasks(client, RuntimeSettings(_env_file=None))
    task = await executor.create(RunInput(goal="file missing"))
    assert client.enqueue_attempts == 1
    queued = client.enqueued[0]
    options, task_id = queued.options, queued.task_id
    goal, max_steps = queued.goal, queued.max_steps
    assert options["workflow_id"] == task_id == str(task.task_id)
    assert options["workflow_name"] == "etl_task" and options["queue_name"] == "etl_tasks"
    assert options["workflow_timeout"] == 180
    assert options["serialization_type"] == WorkflowSerializationFormat.PORTABLE
    assert (goal, max_steps) == ("file missing", 5)


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("ENQUEUED", "pending"),
        ("DELAYED", "pending"),
        ("PENDING", "running"),
        ("ERROR", "error"),
        ("MAX_RECOVERY_ATTEMPTS_EXCEEDED", "error"),
        ("CANCELLED", "interrupted"),
    ],
)
async def test_dbos_status_mapping(native: str, expected: str) -> None:
    client = FakeWorkflowClient()
    client.workflows = [FakeWorkflow(status=native)]
    task = await DBOSTasks(client, RuntimeSettings(_env_file=None)).get(uuid4())
    assert task.status == expected


async def test_dbos_enqueue_failure_is_not_retried() -> None:
    client = FakeWorkflowClient()
    client.enqueue_error = RuntimeError("database unavailable")
    with pytest.raises(ExecutorUnavailable):
        await DBOSTasks(client, RuntimeSettings(_env_file=None)).create(
            RunInput(goal="file missing")
        )
    assert client.enqueue_attempts == 1


async def test_sse_keeps_repeated_nodes_and_stable_sequences() -> None:
    entries = [
        HistoryEntry(role="assistant", node="planning", content="Read then calculate"),
        HistoryEntry(role="assistant", node="acting", content="Read file"),
        HistoryEntry(role="tool", node="observation", content="120 rows, 30 rejected"),
        HistoryEntry(role="assistant", node="acting", content="Calculate percentage"),
        HistoryEntry(role="tool", node="observation", content="25%"),
        HistoryEntry(role="assistant", node="finalization", content="Rejected 25%"),
    ]

    class GrowingExecutor(FakeExecutor):
        polls = 0

        async def history(self, task_id: UUID) -> list[HistoryEntry]:
            self.polls += 1
            # Include an unchanged poll to exercise deduplication, then complete.
            return entries[:2] if self.polls <= 2 else entries

        async def get(self, task_id: UUID) -> Task:
            if self.polls >= 3:
                self.status = "success"
            task = await super().get(task_id)
            if task.result is not None:
                task.result.history = entries
            return task

    executor = GrowingExecutor()
    executor.status = "running"
    service = TaskService(executor)
    initial = await service.get(executor.task_id)
    events = [event async for event in service.events(initial)]
    progress = [event.data for event in events if event.event == "progress"]
    from etl_agent.tasks import TaskProgress

    assert all(isinstance(item, TaskProgress) for item in progress)
    assert [item.sequence for item in progress if isinstance(item, TaskProgress)] == list(
        range(1, 7)
    )
    assert [item.node for item in progress if isinstance(item, TaskProgress)] == [
        entry.node for entry in entries
    ]
    replay = [event async for event in service.events(await service.get(executor.task_id))]
    assert [item.data for item in replay if item.event == "progress"] == progress


async def test_sse_includes_final_history_when_checkpoint_read_lags() -> None:
    class LaggingExecutor(FakeExecutor):
        async def history(self, task_id: UUID) -> list[HistoryEntry]:
            return []

    executor = LaggingExecutor()
    executor.status = "success"
    service = TaskService(executor)
    task = await service.get(executor.task_id)
    events = [event async for event in service.events(task)]
    assert [event.event for event in events] == ["progress", "progress", "progress", "task"]
