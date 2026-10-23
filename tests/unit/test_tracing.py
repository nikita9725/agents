"""Offline LangSmith SDK transport and reconstructable execution scenarios."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client
from langsmith.run_helpers import get_tracing_context
from pydantic import SecretStr
from requests import PreparedRequest, Response, Session

from etl_agent.domain import (
    AgentState,
    DoneCriterion,
    ExecutionPlan,
    MemoryInput,
    Observation,
    RunInput,
    WorkingMemory,
)
from etl_agent.exceptions import TransientPlanningError
from etl_agent.infrastructure.graph import build_graph
from etl_agent.infrastructure.tools import create_tools
from etl_agent.observability import (
    DiagnosticFormatter,
    execution_config,
    execution_tracing,
    request_id,
    run_id,
    task_id,
    thread_id,
)
from etl_agent.settings import Settings
from etl_agent.startup import validate_startup
from tests.fakes import FakePlanner
from tests.unit.test_agent import fast_retries

pytestmark = pytest.mark.unit


class RecordingSession(Session):
    """Injected SDK transport: every HTTP operation stays in memory."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.requests: list[PreparedRequest] = []
        self.fail = fail

    def send(self, request: PreparedRequest, **kwargs: Any) -> Response:
        self.requests.append(request)
        response = Response()
        response.status_code = 503 if self.fail else 200
        response._content = b'{"detail":"unavailable"}' if self.fail else b"{}"
        response.request = request
        response.url = request.url or ""
        return response


class ScenarioPlanner(FakePlanner):
    def __init__(self, scenario: str) -> None:
        super().__init__()
        self.scenario = scenario
        self.acts = 0
        self.memory_attempts = 0

    async def create_plan(self, goal: str) -> ExecutionPlan:
        criteria = [DoneCriterion(description="Calculate 2+2")]
        if self.scenario == "partial":
            criteria.append(DoneCriterion(description="Read missing.log"))
        return ExecutionPlan(steps=[item.description for item in criteria], done_criteria=criteria)

    async def act(self, messages: list[AnyMessage]) -> AIMessage:
        self.acts += 1
        if self.acts > 1:
            return AIMessage(content="Available checks finished")
        calls = [{"name": "calculator", "args": {"expression": "2+2"}, "id": "calc"}]
        if self.scenario == "partial":
            calls.append({"name": "read_local_file", "args": {"path": "missing.log"}, "id": "read"})
        return AIMessage(content="Check available evidence", tool_calls=calls)

    async def update_memory(self, context: MemoryInput) -> WorkingMemory:
        self.memory_attempts += 1
        if self.scenario == "error":
            raise TransientPlanningError("Memory provider temporarily unavailable")
        successful = [item for item in context.results if item.message.status == "success"]
        return WorkingMemory(
            observations=[Observation(summary="2+2=4", evidence=[successful[0].id])],
            done_criteria=[
                item.model_copy(
                    update={
                        "satisfied": index == 0,
                        "evidence": [successful[0].id] if index == 0 else [],
                    }
                )
                for index, item in enumerate(context.done_criteria)
            ],
        )


def settings(*, enabled: bool = False) -> Settings:
    return Settings(
        _env_file=None,
        llm_model="test-model",
        llm_api_key=SecretStr("test-llm-key"),
        langsmith_tracing=enabled,
        langsmith_api_key=SecretStr("test-trace-key"),
        langsmith_project="offline-day6",
        langsmith_endpoint="https://tracing.invalid",
    )


def sdk_client(config: Settings, transport: RecordingSession) -> Client:
    return Client(
        api_url=config.langsmith_endpoint,
        api_key=config.langsmith_api_key.get_secret_value() if config.langsmith_api_key else None,
        session=transport,
        auto_batch_tracing=False,
        info={},
    )


async def test_disabled_tracing_never_constructs_client() -> None:
    def forbidden(config: Settings) -> Client:
        raise AssertionError("disabled tracing must not construct a client")

    async with execution_tracing(settings(), client_factory=forbidden):
        assert get_tracing_context()["enabled"] is False
        await build_graph(FakePlanner()).ainvoke(RunInput(goal="offline"))


@pytest.mark.parametrize("scenario", ["success", "partial", "error"])
async def test_scenario_reconstructs_tree_state_and_failures(
    scenario: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    transport = RecordingSession()
    config = execution_config("task-" + scenario, "request-" + scenario, 5)
    planner = ScenarioPlanner(scenario)
    saver = InMemorySaver()
    graph = build_graph(
        planner, tools=create_tools(tmp_path), checkpointer=saver, retry_policy=fast_retries()
    )
    with caplog.at_level(logging.INFO):
        async with execution_tracing(
            settings(enabled=True), client_factory=lambda config: sdk_client(config, transport)
        ):
            if scenario == "error":
                with pytest.raises(TransientPlanningError):
                    await graph.ainvoke(RunInput(goal="Inspect"), config)
            else:
                state = AgentState.model_validate(
                    await graph.ainvoke(RunInput(goal="Inspect"), config)
                )
                assert state.stop_reason == (
                    "criteria_met" if scenario == "success" else "actor_finished"
                )
                assert [item.message.status for item in state.tool_results] == (
                    ["success"] if scenario == "success" else ["success", "error"]
                )
                if scenario == "partial":
                    assert state.done_criteria[0].satisfied and not state.done_criteria[1].satisfied
                    assert "Не подтверждено: Read missing.log" in str(state.final_answer)
                assert state.history[-1].state_delta
    # Real SDK create/update requests, including inherited graph/tool parent relationships.
    creates = [
        json.loads(bytes(request.body) if isinstance(request.body, bytes) else str(request.body))
        for request in transport.requests
        if request.method == "POST"
    ]
    updates = [
        json.loads(bytes(request.body) if isinstance(request.body, bytes) else str(request.body))
        for request in transport.requests
        if request.method == "PATCH"
    ]
    roots = [item for item in creates if not item.get("parent_run_id")]
    assert len(roots) == 1 and roots[0]["name"] == "etl-agent"
    assert roots[0]["id"] == str(config["run_id"])
    assert roots[0]["session_name"] == "offline-day6"
    metadata = config["metadata"]
    assert all(item["extra"]["metadata"]["task_id"] == metadata["task_id"] for item in creates)
    assert all(item["extra"]["metadata"]["thread_id"] == metadata["thread_id"] for item in creates)
    identifiers = {item["id"] for item in creates}
    assert all(
        item["parent_run_id"] in identifiers for item in creates if item.get("parent_run_id")
    )
    assert any(item["name"] == "calculator" and item["run_type"] == "tool" for item in creates)
    assert any(item["name"] == "observation" for item in creates)
    assert any(bool(item.get("error")) for item in updates) == (scenario == "error")
    assert caplog.text.count("event=tool.started name=calculator") == 1
    assert "event=action.chosen kind=tools" in caplog.text
    assert "tool_call_id=calc result_id=result_1 status=success" in caplog.text
    assert "event=state.updated node=tools" in caplog.text
    if scenario == "error":
        assert planner.memory_attempts == 3
        assert caplog.text.count("event=node.failed node=observation") == 3
        assert "event=node.completed node=observation" not in caplog.text
        snapshot = await graph.aget_state(config)
        assert len(snapshot.values["tool_results"]) == 1
        assert not snapshot.values.get("observations", [])
    else:
        assert "event=node.chosen node=observation" in caplog.text
        assert "event=node.completed node=finalization" in caplog.text


async def test_unavailable_tracing_does_not_fail_execution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unavailable(config: Settings) -> Client:
        raise RuntimeError("private transport detail")

    async with execution_tracing(settings(enabled=True), client_factory=unavailable):
        result = await build_graph(FakePlanner()).ainvoke(RunInput(goal="Inspect"))
    assert result["final_answer"]
    assert "event=tracing.unavailable error_type=RuntimeError" in caplog.text
    assert "private transport detail" not in caplog.text


@pytest.mark.parametrize("key", [None, "", "replace-me"])
def test_enabled_tracing_requires_key(key: str | None) -> None:
    config = settings(enabled=True).model_copy(
        update={"langsmith_api_key": SecretStr(key) if key is not None else None}
    )
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        validate_startup(config)


def test_enabled_tracing_requires_project() -> None:
    with pytest.raises(ValueError, match="LANGSMITH_PROJECT"):
        validate_startup(settings(enabled=True).model_copy(update={"langsmith_project": " "}))


def test_recovery_keeps_identity_but_changes_execution() -> None:
    first, second = execution_config("task", "request", 5), execution_config("task", "request", 5)
    assert first["run_id"] != second["run_id"]
    assert first["metadata"] == second["metadata"]
    assert first["configurable"] == second["configurable"]


async def test_parallel_tasks_keep_correlated_logs(tmp_path: Path) -> None:
    records: list[tuple[str, str, str, str]] = []

    class CorrelatedPlanner(ScenarioPlanner):
        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            await asyncio.sleep(0)
            records.append(
                (str(task_id.get()), str(thread_id.get()), str(request_id.get()), str(run_id.get()))
            )
            return await super().act(messages)

    async def execute(identifier: str) -> None:
        await build_graph(CorrelatedPlanner("success"), tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal=identifier), execution_config(identifier, "request-" + identifier, 5)
        )

    await asyncio.gather(execute("one"), execute("two"))
    assert {item[:3] for item in records} == {
        ("one", "one", "request-one"),
        ("two", "two", "request-two"),
    }
    assert len({item[3] for item in records}) == 2
    assert task_id.get() is None and request_id.get() is None


async def test_delivery_failure_preserves_result() -> None:
    transport = RecordingSession(fail=True)
    async with execution_tracing(
        settings(enabled=True), client_factory=lambda config: sdk_client(config, transport)
    ):
        result = await build_graph(FakePlanner()).ainvoke(RunInput(goal="Inspect"))
    assert result["final_answer"] and transport.requests


def test_formatter_masks_explicit_trace_credentials() -> None:
    from etl_agent.observability import redacted_secrets

    redacted_secrets.add("specific-trace-secret")
    try:
        record = logging.LogRecord(
            "test", logging.WARNING, __file__, 1, "Bearer specific-trace-secret", (), None
        )
        assert "specific-trace-secret" not in DiagnosticFormatter("%(message)s").format(record)
    finally:
        redacted_secrets.discard("specific-trace-secret")


async def test_cancelled_tool_has_cancel_event_and_no_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from langchain_core.tools import tool

    started = asyncio.Event()

    @tool
    async def calculator(expression: str) -> str:
        """Wait until the graph is cancelled."""
        started.set()
        await asyncio.Event().wait()
        return expression

    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(
            build_graph(ScenarioPlanner("success"), tools=[calculator]).ainvoke(
                RunInput(goal="Inspect"), execution_config("cancelled-task", None, 5)
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert "event=node.cancelled node=tools" in caplog.text
    assert "event=node.failed node=tools" not in caplog.text
    assert "event=node.completed node=tools" not in caplog.text
    assert "event=tool.failed" not in caplog.text


async def test_resume_has_distinct_root_and_keeps_saved_tool(tmp_path: Path) -> None:
    transport = RecordingSession()
    planner = ScenarioPlanner("success")
    saver = InMemorySaver()
    graph = build_graph(planner, tools=create_tools(tmp_path), checkpointer=saver)
    first = execution_config("resumed-task", "original-request", 5)
    second = execution_config("resumed-task", "original-request", 5)
    async with execution_tracing(
        settings(enabled=True), client_factory=lambda config: sdk_client(config, transport)
    ):
        await graph.ainvoke(RunInput(goal="Inspect"), first, interrupt_after=["tools"])
        result = await graph.ainvoke(None, second)
    assert result["stop_reason"] == "criteria_met"
    creates = [
        json.loads(bytes(request.body) if isinstance(request.body, bytes) else str(request.body))
        for request in transport.requests
        if request.method == "POST"
    ]
    roots = [item for item in creates if not item.get("parent_run_id")]
    assert {item["id"] for item in roots} == {str(first["run_id"]), str(second["run_id"])}
    assert {item["extra"]["metadata"]["thread_id"] for item in roots} == {"resumed-task"}
    assert sum(item["name"] == "calculator" and item["run_type"] == "tool" for item in creates) == 1


async def test_flush_failure_preserves_original_graph_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ClosingFailureClient(Client):
        def close(self, timeout: float | None = None) -> None:
            super().close(timeout=timeout)
            raise RuntimeError("private flush failure")

    client = ClosingFailureClient(api_url="https://unused.invalid", api_key="unit-key", info={})
    with pytest.raises(ValueError, match="graph failure"):
        async with execution_tracing(settings(enabled=True), client_factory=lambda config: client):
            raise ValueError("graph failure")
    assert "event=tracing.flush_failed error_type=RuntimeError" in caplog.text
    assert "private flush failure" not in caplog.text


def test_production_client_uses_explicit_settings() -> None:
    from etl_agent.observability import create_tracing_client

    config = settings(enabled=True)
    client = create_tracing_client(config, session=RecordingSession())
    try:
        assert client.api_url == config.langsmith_endpoint
        assert client.api_key == "test-trace-key"
    finally:
        client.close(timeout=2)


def test_dotenv_drives_tracing_without_exporting_environment(tmp_path: Path) -> None:
    import subprocess
    import sys

    (tmp_path / ".env").write_text(
        "LLM_MODEL=test-model\nLLM_API_KEY=test-key\nLANGSMITH_TRACING=true\n"
        "LANGSMITH_PROJECT=dotenv-project\nLANGSMITH_API_KEY=dotenv-key\n"
        "LANGSMITH_ENDPOINT=https://dotenv.invalid\n"
    )
    probe = """
import json
import os
from etl_agent.settings import Settings
from etl_agent.startup import validate_startup
config = Settings()
validate_startup(config)
print(json.dumps({"enabled": config.langsmith_tracing,
                  "project": config.langsmith_project,
                  "endpoint": config.langsmith_endpoint,
                  "exported": "LANGSMITH_TRACING" in os.environ}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env={},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "enabled": True,
        "project": "dotenv-project",
        "endpoint": "https://dotenv.invalid",
        "exported": False,
    }
