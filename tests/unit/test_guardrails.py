"""Детерминированные сценарии ошибок, циклов, лимитов и восстановления для дня 5."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from etl_agent.domain import (
    AgentState,
    DoneCriterion,
    ExecutionPlan,
    MemoryInput,
    RunInput,
    WorkingMemory,
)
from etl_agent.exceptions import TransientPlanningError
from etl_agent.infrastructure.graph import build_graph
from etl_agent.infrastructure.tools import create_tools
from tests.unit.test_agent import fast_retries
from tests.unit.test_tool_graph import ToolPlanner

pytestmark = pytest.mark.unit


def calculate(expression: str, identifier: str) -> ToolCall:
    return {"name": "calculator", "args": {"expression": expression}, "id": identifier}


class ScenarioPlanner(ToolPlanner):
    def __init__(self, batches: list[list[ToolCall]]) -> None:
        super().__init__()
        self.batches = batches

    async def act(self, messages: list[AnyMessage]) -> AIMessage:
        self.inputs.append(list(messages))
        index = len(self.inputs) - 1
        if index >= len(self.batches):
            return AIMessage(content="Диагностика завершена")
        return AIMessage(content="", tool_calls=self.batches[index])


async def test_errors_with_changing_arguments_stop_and_explain_trace(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    planner = ScenarioPlanner([[calculate("1/0", "first")], [calculate("2/0", "second")]])
    with caplog.at_level(logging.INFO):
        state = AgentState.model_validate(
            await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
                RunInput(goal="Calculate", max_steps=2)
            )
        )
    assert state.stop_reason == "no_progress" and state.current_step == 2
    assert state.consecutive_no_progress == 2
    assert all(item.message.status == "error" for item in state.tool_results)
    assert "два пакета подряд" in str(state.final_answer)
    assert any("no_progress" in str(message.content) for message in planner.final_inputs)
    assert "stop_reason=no_progress current_step=2 max_steps=2" in caplog.text
    deltas = [entry.state_delta for entry in state.history if entry.node == "observation"]
    assert [delta["consecutive_no_progress"] for delta in deltas if delta] == [1, 2]
    assert state.history[-1].state_delta
    assert state.history[-1].state_delta["stop_reason"] == "no_progress"
    assert AgentState.model_validate_json(state.model_dump_json()) == state


async def test_correction_and_mixed_batch_reset_no_progress(tmp_path: Path) -> None:
    planner = ScenarioPlanner(
        [
            [calculate("1/0", "bad1")],
            [calculate("2+2", "good"), calculate("2/0", "bad2")],
            [calculate("3/0", "bad3")],
        ]
    )
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(RunInput(goal="Inspect"))
    )
    assert state.stop_reason == "actor_finished" and state.current_step == 4
    assert state.consecutive_no_progress == 1
    deltas = [entry.state_delta for entry in state.history if entry.node == "observation"]
    assert [delta["consecutive_no_progress"] for delta in deltas if delta] == [1, 0, 1]
    tool_deltas = [entry.state_delta for entry in state.history if entry.node == "tools"]
    assert [delta["new_successful_results"] for delta in tool_deltas if delta] == [0, 1, 0]


@pytest.mark.parametrize("expression", ["1+1", "1/0"])
async def test_repeated_success_or_error_executes_once(expression: str, tmp_path: Path) -> None:
    planner = ScenarioPlanner([[calculate(expression, "first")], [calculate(expression, "repeat")]])
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal="Inspect", max_steps=2)
        )
    )
    assert state.stop_reason == "no_new_results" and len(state.tool_results) == 1
    messages = [message for message in state.messages if isinstance(message, ToolMessage)]
    assert messages[-1].additional_kwargs["reused"] is True
    assert messages[-1].tool_call_id == "repeat"
    assert state.consecutive_no_progress == (2 if expression == "1/0" else 1)


async def test_cycle_a_b_a_reuses_original_success(tmp_path: Path) -> None:
    planner = ScenarioPlanner(
        [[calculate("1+1", "a")], [calculate("2+2", "b")], [calculate("1+1", "a_again")]]
    )
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(RunInput(goal="Inspect"))
    )
    assert state.stop_reason == "no_new_results" and state.current_step == 3
    assert len(state.tool_results) == 2
    assert state.messages[-1].additional_kwargs == {"result_id": "result_1", "reused": True}


async def test_cached_success_and_new_error_are_not_progress(tmp_path: Path) -> None:
    planner = ScenarioPlanner(
        [
            [calculate("1+1", "success")],
            [calculate("1+1", "cached1"), calculate("1/0", "bad1")],
            [calculate("1+1", "cached2"), calculate("2/0", "bad2")],
        ]
    )
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(RunInput(goal="Inspect"))
    )
    assert state.stop_reason == "no_progress" and state.consecutive_no_progress == 2


@pytest.mark.parametrize("max_steps", [1, 12])
async def test_budget_runs_last_batch_without_recursion_failure(
    max_steps: int, tmp_path: Path
) -> None:
    # Different successful calls may return identical content; that is still progress.
    planner = ScenarioPlanner(
        [[calculate(f"{index}-{index}", f"call_{index}")] for index in range(max_steps)]
    )
    config: RunnableConfig = {"recursion_limit": max(25, max_steps * 3 + 5)}
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal="Inspect", max_steps=max_steps), config
        )
    )
    assert state.stop_reason == "step_limit" and state.current_step == max_steps
    assert len(state.tool_results) == max_steps and state.consecutive_no_progress == 0
    assert len(planner.inputs) == max_steps
    assert any(isinstance(message, ToolMessage) for message in planner.final_inputs)


async def test_exhausted_checkpoint_skips_actor_and_tools(tmp_path: Path) -> None:
    planner = ScenarioPlanner([[calculate("1+1", "first")]])
    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(planner, tools=create_tools(tmp_path), checkpointer=saver)
    await graph.ainvoke(RunInput(goal="Inspect", max_steps=1), config, interrupt_after=["planning"])
    await graph.aupdate_state(config, {"current_step": 1})
    state = AgentState.model_validate(await graph.ainvoke(None, config))
    assert state.stop_reason == "step_limit" and state.current_step == 1
    assert not planner.inputs and not state.tool_results


@pytest.mark.parametrize("boundary", ["tools", "observation"])
async def test_error_streak_survives_resume_without_double_counting(boundary: str) -> None:
    calls = 0

    @tool
    def calculator(expression: str) -> str:
        """Вызывать ошибку и подсчитывать фактические выполнения инструмента."""
        nonlocal calls
        calls += 1
        raise RuntimeError("private implementation detail")

    planner = ScenarioPlanner([[calculate("1/0", "first")], [calculate("2/0", "second")]])
    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(planner, tools=[calculator], checkpointer=saver)
    await graph.ainvoke(RunInput(goal="Inspect"), config, interrupt_after=[boundary])
    snapshot = await graph.aget_state(config)
    assert snapshot.values.get("consecutive_no_progress", 0) == (boundary == "observation")
    state = AgentState.model_validate(
        await build_graph(planner, tools=[calculator], checkpointer=saver).ainvoke(None, config)
    )
    assert calls == 2 and len(state.tool_results) == 2
    assert state.stop_reason == "no_progress" and state.consecutive_no_progress == 2


async def test_injected_tool_validation_and_exception_allow_correction() -> None:
    calls: list[str] = []

    @tool
    def calculator(expression: str) -> str:
        """Вызывать ошибку для некорректного выражения и успешно выполнять исправленный вызов."""
        calls.append(expression)
        if expression == "broken":
            raise RuntimeError("private detail")
        return "2"

    planner = ScenarioPlanner(
        [
            [
                {"name": "calculator", "args": {}, "id": "bad_args"},
                {"name": "unknown", "args": {}, "id": "unknown"},
                calculate("broken", "broken"),
                calculate("1+1", "good"),
            ],
            [calculate("2", "corrected")],
        ]
    )
    state = AgentState.model_validate(
        await build_graph(planner, tools=[calculator]).ainvoke(RunInput(goal="Inspect"))
    )
    assert [item.message.status for item in state.tool_results] == [
        "error",
        "error",
        "error",
        "success",
        "success",
    ]
    assert calls == ["broken", "1+1", "2"]
    assert "private detail" not in state.model_dump_json()
    assert state.stop_reason == "actor_finished" and state.consecutive_no_progress == 0


async def test_tool_cancellation_propagates() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @tool
    async def calculator(expression: str) -> str:
        """Ожидать внешней отмены выполнения."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return expression

    planner = ScenarioPlanner([[calculate("1+1", "wait")]])
    task = asyncio.create_task(
        build_graph(planner, tools=[calculator]).ainvoke(RunInput(goal="Inspect"))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set() and not planner.final_inputs


@pytest.mark.parametrize("repeat", [False, True])
async def test_criteria_take_precedence_over_other_stops(repeat: bool, tmp_path: Path) -> None:
    class CompletingPlanner(ScenarioPlanner):
        async def create_plan(self, goal: str) -> ExecutionPlan:
            return ExecutionPlan(
                steps=["Calculate"], done_criteria=[DoneCriterion(description="Calculated")]
            )

        async def update_memory(self, context: MemoryInput) -> WorkingMemory:
            return WorkingMemory(
                done_criteria=[
                    DoneCriterion(
                        description="Calculated",
                        satisfied=len(self.inputs) == 3,
                        evidence=["result_1"] if len(self.inputs) == 3 else [],
                    )
                ]
            )

    planner = CompletingPlanner(
        [
            [calculate("1+1", "success")],
            [calculate("1/0", "bad")],
            [calculate("1/0" if repeat else "2/0", "last")],
        ]
    )
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal="Inspect", max_steps=3)
        )
    )
    assert state.stop_reason == "criteria_met" and state.consecutive_no_progress == 2
    assert "Критерии диагностики выполнены" in str(state.final_answer)


async def test_observation_retry_does_not_double_count_error_batch(tmp_path: Path) -> None:
    class RetryingMemory(ScenarioPlanner):
        attempts = 0

        async def update_memory(self, context: MemoryInput) -> WorkingMemory:
            self.attempts += 1
            if self.attempts == 2:
                raise TransientPlanningError("Retry memory update")
            return await super().update_memory(context)

    planner = RetryingMemory([[calculate("1/0", "first")], [calculate("2/0", "second")]])
    state = AgentState.model_validate(
        await build_graph(
            planner, tools=create_tools(tmp_path), retry_policy=fast_retries()
        ).ainvoke(RunInput(goal="Inspect"))
    )
    assert planner.attempts == 3 and state.consecutive_no_progress == 2
    assert state.stop_reason == "no_progress" and len(state.tool_results) == 2
    assert len([entry for entry in state.history if entry.node == "observation"]) == 2


async def test_native_tool_interrupt_is_preserved() -> None:
    @tool
    def calculator(expression: str) -> str:
        """Приостанавливать выполнение штатным механизмом прерываний графа."""
        return str(interrupt("Await external input"))

    planner = ScenarioPlanner([[calculate("1+1", "paused")]])
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(planner, tools=[calculator], checkpointer=InMemorySaver())
    paused = await graph.ainvoke(RunInput(goal="Inspect"), config)
    assert paused["__interrupt__"]
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("tools",) and not snapshot.values.get("tool_results")
    state = AgentState.model_validate(await graph.ainvoke(Command(resume="2"), config))
    assert state.stop_reason == "actor_finished"
    assert state.tool_results[0].message.status == "success"
    assert state.tool_results[0].message.content == "2"
