import asyncio
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeTimeoutError
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from etl_agent.domain import AgentState, NextAction, RunInput
from etl_agent.exceptions import PlanningError, PlanningTimeout, TransientPlanningError
from etl_agent.infrastructure.graph import build_graph, retry_planning
from tests.fakes import FakePlanner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("goal", ["", " ", "\n\t"])
def test_empty_goal_rejected(goal: str) -> None:
    with pytest.raises(ValidationError):
        RunInput(goal=goal)


def test_input_cannot_override_internal_state() -> None:
    with pytest.raises(ValidationError):
        RunInput.model_validate({"goal": "missing file", "current_step": 100})


@pytest.mark.parametrize("max_steps", [0, -1])
def test_invalid_budget_rejected(max_steps: int) -> None:
    with pytest.raises(ValidationError):
        RunInput(goal="failed", max_steps=max_steps)


async def test_graph_preserves_history_and_checkpoints() -> None:
    planner = FakePlanner()
    graph = build_graph(planner, checkpointer=InMemorySaver())
    run_config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    result = await graph.ainvoke(RunInput(goal="missing file"), run_config, durability="sync")
    assert result["current_step"] == 1
    assert result["final_answer"]
    assert [item.node for item in result["history"]] == ["input", "planning", "finalization"]
    assert planner.calls == ["missing file"]
    assert len([item async for item in graph.aget_state_history(run_config)]) >= 3


class ScriptedPlanner(FakePlanner):
    def __init__(self, errors: list[Exception]) -> None:
        super().__init__()
        self._errors = errors
        self.attempts = 0

    async def propose(self, goal: str) -> NextAction:
        self.attempts += 1
        if self._errors:
            raise self._errors.pop(0)
        return await super().propose(goal)


def fast_retries() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=3, initial_interval=0.001, jitter=False, retry_on=retry_planning
    )


async def test_transient_error_retries_without_duplicating_history() -> None:
    planner = ScriptedPlanner([TransientPlanningError("busy"), PlanningTimeout("slow")])
    result = await build_graph(planner, retry_policy=fast_retries()).ainvoke(
        RunInput(goal="file missing")
    )
    assert planner.attempts == 3
    assert result["current_step"] == 1
    assert len(result["history"]) == 3


async def test_retries_are_bounded_and_permanent_failure_is_not_retried() -> None:
    for error_type, expected in [(TransientPlanningError, 3), (PlanningError, 1)]:
        planner = ScriptedPlanner([error_type("failed") for _ in range(4)])
        graph = build_graph(planner, retry_policy=fast_retries(), checkpointer=InMemorySaver())
        run_config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        with pytest.raises(error_type):
            await graph.ainvoke(RunInput(goal="file missing"), run_config)
        assert planner.attempts == expected
        snapshot = await graph.aget_state(run_config)
        assert snapshot.values.get("final_answer") is None
        assert snapshot.values.get("current_step", 0) == 0


class HangingPlanner(FakePlanner):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = 0

    async def propose(self, goal: str) -> NextAction:
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled += 1
        return await super().propose(goal)


async def test_native_node_timeout_cancels_each_attempt() -> None:
    planner = HangingPlanner()
    graph = build_graph(planner, retry_policy=fast_retries(), planning_timeout=0.01)
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke(RunInput(goal="missing file"))
    assert planner.cancelled == 3


async def test_external_cancellation_is_not_retried() -> None:
    planner = HangingPlanner()
    task = asyncio.create_task(build_graph(planner).ainvoke(RunInput(goal="missing file")))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert planner.cancelled == 1


def test_independent_history_defaults() -> None:
    assert AgentState(goal="one").history is not AgentState(goal="two").history


async def test_concurrent_graphs_keep_native_context_isolated() -> None:
    from etl_agent.observability import run_id, thread_id

    barrier = asyncio.Event()
    contexts: list[tuple[str | None, str | None]] = []

    class ConcurrentPlanner(FakePlanner):
        async def propose(self, goal: str) -> NextAction:
            before = (thread_id.get(), run_id.get())
            contexts.append(before)
            if len(contexts) == 2:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), 2)
            assert (thread_id.get(), run_id.get()) == before
            return await super().propose(goal)

    graph = build_graph(ConcurrentPlanner(), checkpointer=InMemorySaver())
    identifiers = [str(uuid4()), str(uuid4())]
    await asyncio.gather(
        *[
            graph.ainvoke(
                RunInput(goal="missing file"), {"configurable": {"thread_id": identifier}}
            )
            for identifier in identifiers
        ]
    )
    assert {context[0] for context in contexts} == set(identifiers)
    assert thread_id.get() is None
    assert run_id.get() is None
