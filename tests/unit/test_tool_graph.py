import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from etl_agent.domain import AgentState, FinalResponse, NextAction, RunInput
from etl_agent.infrastructure.graph import build_graph
from etl_agent.infrastructure.tools import create_tools
from tests.fakes import FakePlanner

pytestmark = pytest.mark.unit


class ToolPlanner(FakePlanner):
    def __init__(self, *, repeat: bool = False) -> None:
        super().__init__()
        self.inputs: list[list[AnyMessage]] = []
        self.final_inputs: list[AnyMessage] = []
        self.repeat = repeat

    async def act(self, messages: list[AnyMessage]) -> AIMessage:
        self.inputs.append(list(messages))
        if len(self.inputs) == 1 or self.repeat:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "120-90"},
                        "id": f"calc_{len(self.inputs)}",
                    },
                    {
                        "name": "read_local_file",
                        "args": {"path": "etl.log"},
                        "id": f"read_{len(self.inputs)}",
                    },
                ],
            )
        assert isinstance(messages[-1], ToolMessage)
        return AIMessage(content="30 rows missing")

    async def finalize(self, messages: list[AnyMessage]) -> FinalResponse:
        self.final_inputs = list(messages)
        return FinalResponse(
            final_answer="30 rows missing",
            next_action=NextAction(
                action="Check rejected rows", rationale="Observed 30 missing rows"
            ),
        )


async def test_observations_loop_serialization_and_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "etl.log").write_text("ERROR 30 rows")
    planner = ToolPlanner()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(planner, tools=create_tools(tmp_path), checkpointer=InMemorySaver())
    with caplog.at_level(logging.INFO):
        result = await graph.ainvoke(RunInput(goal="Inspect ETL"), config)
    state = AgentState.model_validate(result)
    assert state.current_step == 2
    assert state.final_answer == "30 rows missing" and state.next_action
    observations = [message for message in planner.inputs[1] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in observations] == ["calc_1", "read_1"]
    assert json.loads(str(observations[0].content)) == {"result": 30.0}
    assert "ERROR 30 rows" in str(observations[1].content)
    assert AgentState.model_validate_json(state.model_dump_json()) == state
    assert "event=tool.started" in caplog.text and "event=tool.completed" in caplog.text
    assert "tool_call_id=calc_1" in caplog.text and "result=" in caplog.text
    assert [entry.node for entry in state.history] == [
        "input",
        "planning",
        "acting",
        "tools",
        "observation",
        "observation",
        "acting",
        "finalization",
    ]


async def test_budget_executes_last_calls_and_finalizes(tmp_path: Path) -> None:
    planner = ToolPlanner(repeat=True)
    result = await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
        RunInput(goal="Inspect", max_steps=1)
    )
    assert len(planner.inputs) == 1
    assert (
        len([message for message in planner.final_inputs if isinstance(message, ToolMessage)]) == 2
    )
    assert "исчерпан" in result["final_answer"]


async def test_bad_arguments_and_unknown_tool_can_be_corrected(tmp_path: Path) -> None:
    class CorrectingPlanner(ToolPlanner):
        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            self.inputs.append(list(messages))
            if len(self.inputs) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "args": {}, "id": "bad"},
                        {"name": "unknown", "args": {}, "id": "unknown"},
                    ],
                )
            if len(self.inputs) == 2:
                assert all(
                    message.status == "error"
                    for message in messages[-2:]
                    if isinstance(message, ToolMessage)
                )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "args": {"expression": "1+1"}, "id": "good"}
                    ],
                )
            assert isinstance(messages[-1], ToolMessage) and messages[-1].status == "success"
            return AIMessage(content="2")

    planner = CorrectingPlanner()
    result = await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
        RunInput(goal="Calculate")
    )
    assert result["current_step"] == 3


@pytest.mark.parametrize("boundary", ["planning", "acting", "tools", "observation"])
async def test_resume_saved_tool_boundary_without_reexecution(boundary: str) -> None:
    calls = 0

    @tool
    def calculator(expression: str) -> str:
        """Подсчитывать фактические выполнения инструмента."""
        nonlocal calls
        calls += 1
        return '{"result":30}'

    planner = ToolPlanner()
    tools = [calculator]
    saver = InMemorySaver()
    graph = build_graph(planner, tools=tools, checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    await graph.ainvoke(RunInput(goal="Inspect"), config, interrupt_after=[boundary])
    assert calls == (0 if boundary in {"planning", "acting"} else 1)
    snapshot = await graph.aget_state(config)
    expected_next = {
        "planning": "acting",
        "acting": "tools",
        "tools": "observation",
        "observation": "acting",
    }
    assert snapshot.next == (expected_next[boundary],)
    if boundary == "tools":
        assert len(snapshot.values["pending_observations"]) == 2
        assert not any(isinstance(message, ToolMessage) for message in snapshot.values["messages"])
    if boundary == "observation":
        assert snapshot.values["pending_observations"] == []
    restored = build_graph(planner, tools=tools, checkpointer=saver)
    result = await restored.ainvoke(None, config)
    assert result["final_answer"] and calls == 1
    assert planner.calls == ["Inspect"]
    assert result["plan"].steps


async def test_unexpected_tool_bug_becomes_safe_observation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @tool
    def calculator(expression: str) -> str:
        """Вызывать исключение для проверки его преобразования в безопасный результат ошибки."""
        raise RuntimeError("implementation bug")

    with caplog.at_level(logging.INFO):
        result = await build_graph(ToolPlanner(), tools=[calculator]).ainvoke(
            RunInput(goal="Inspect")
        )
    state = AgentState.model_validate(result)
    failed = state.tool_results[0]
    assert failed.message.status == "error"
    assert failed.message.tool_call_id == "calc_1" and failed.message.name == "calculator"
    assert "implementation bug" not in state.model_dump_json()
    assert "implementation bug" in caplog.text and "event=tool.failed" in caplog.text
    assert state.stop_reason == "actor_finished" and state.final_answer


async def test_finalization_retries_without_repeating_tools(tmp_path: Path) -> None:
    from etl_agent.exceptions import TransientPlanningError
    from tests.unit.test_agent import fast_retries

    class RetryingPlanner(ToolPlanner):
        attempts = 0

        async def finalize(self, messages: list[AnyMessage]) -> FinalResponse:
            self.attempts += 1
            if self.attempts == 1:
                raise TransientPlanningError("busy")
            return await super().finalize(messages)

    planner = RetryingPlanner()
    result = await build_graph(
        planner, tools=create_tools(tmp_path), retry_policy=fast_retries()
    ).ainvoke(RunInput(goal="Inspect"))
    assert planner.attempts == 2 and len(planner.inputs) == 2
    assert len([entry for entry in result["history"] if entry.node == "observation"]) == 2
    assert len([entry for entry in result["history"] if entry.node == "finalization"]) == 1


def test_old_completed_result_remains_readable() -> None:
    state = AgentState.model_validate(
        {
            "goal": "old task",
            "max_steps": 1,
            "current_step": 1,
            "next_action": {"action": "Check logs", "rationale": "Need evidence"},
            "final_answer": "Check logs",
            "history": [],
        }
    )
    assert state.messages == [] and state.final_answer
    assert state.plan is None and state.pending_observations == []
    assert state.consecutive_no_progress == 0


async def test_plan_read_observation_then_dependent_calculation(tmp_path: Path) -> None:
    from etl_agent.domain import ExecutionPlan

    (tmp_path / "etl.log").write_text("expected_rows=120 rejected_rows=30")

    class SequentialPlanner(ToolPlanner):
        plans = 0

        async def create_plan(self, goal: str) -> ExecutionPlan:
            self.plans += 1
            return ExecutionPlan(steps=["Прочитать etl.log", "Вычислить процент отклонений"])

        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            self.inputs.append(list(messages))
            assert "Вычислить процент отклонений" in str(messages[1].content)
            if len(self.inputs) == 1:
                return AIMessage(
                    content="Читаю исходные числа",
                    tool_calls=[
                        {
                            "name": "read_local_file",
                            "args": {"path": "etl.log"},
                            "id": "read",
                        }
                    ],
                )
            if len(self.inputs) == 2:
                observation = messages[-1]
                assert isinstance(observation, ToolMessage)
                content = json.loads(str(observation.content))["content"]
                counts = dict(item.split("=") for item in content.split())
                return AIMessage(
                    content="Считаю по полученным данным",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "id": "calc",
                            "args": {
                                "expression": f"{counts['rejected_rows']}/{counts['expected_rows']}*100"
                            },
                        }
                    ],
                )
            assert isinstance(messages[-1], ToolMessage)
            assert json.loads(str(messages[-1].content))["result"] == 25
            return AIMessage(content="25%")

        async def finalize(self, messages: list[AnyMessage]) -> FinalResponse:
            return FinalResponse(
                final_answer="Отклонено 25%",
                next_action=NextAction(
                    action="Проверить отклонённые строки",
                    rationale="Отклонено 25%",
                ),
            )

    planner = SequentialPlanner()
    state = AgentState.model_validate(
        await build_graph(
            planner,
            tools=create_tools(tmp_path),
        ).ainvoke(RunInput(goal="Прочитай лог и вычисли процент"))
    )
    assert planner.plans == 1
    assert state.plan and len(state.plan.steps) == 2
    assert state.pending_observations == []
    assert state.final_answer == "Отклонено 25%"
    assert [entry.node for entry in state.history] == [
        "input",
        "planning",
        "acting",
        "tools",
        "observation",
        "acting",
        "tools",
        "observation",
        "acting",
        "finalization",
    ]
    assert '"name": "read_local_file"' in state.history[2].content
    assert '"name": "calculator"' in state.history[5].content


async def test_actor_retry_does_not_repeat_plan_or_duplicate_history() -> None:
    from etl_agent.exceptions import TransientPlanningError
    from tests.unit.test_agent import fast_retries

    class RetryingActor(FakePlanner):
        attempts = 0

        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            self.attempts += 1
            if self.attempts == 1:
                raise TransientPlanningError("busy")
            return await super().act(messages)

    planner = RetryingActor()
    state = AgentState.model_validate(
        await build_graph(
            planner,
            retry_policy=fast_retries(),
        ).ainvoke(RunInput(goal="Inspect"))
    )
    assert planner.calls == ["Inspect"]
    assert planner.attempts == 2 and state.current_step == 1
    assert [entry.node for entry in state.history] == [
        "input",
        "planning",
        "acting",
        "finalization",
    ]
