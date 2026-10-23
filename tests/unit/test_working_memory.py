"""Working memory changes decisions, survives resume, and prevents duplicate work."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from etl_agent.domain import (
    AgentState,
    DoneCriterion,
    ExecutionPlan,
    FinalResponse,
    MemoryInput,
    NextAction,
    Observation,
    RunInput,
    WorkingMemory,
)
from etl_agent.exceptions import PlanningError, TransientPlanningError
from etl_agent.infrastructure.graph import build_graph
from etl_agent.infrastructure.tools import create_tools
from etl_agent.memory import reject_failed_evidence, validate_memory
from tests.fakes import FakePlanner
from tests.unit.test_agent import fast_retries
from tests.unit.test_tool_graph import ToolPlanner

pytestmark = pytest.mark.unit


class MemoryPlanner(FakePlanner):
    def __init__(self) -> None:
        super().__init__()
        self.actors = 0
        self.updates: list[MemoryInput] = []
        self.final_messages: list[AnyMessage] = []

    async def create_plan(self, goal: str) -> ExecutionPlan:
        return ExecutionPlan(
            steps=["Прочитать лог", "Проверить арифметику", "Вычислить процент"],
            done_criteria=[DoneCriterion(description="Вычислен процент отклонений")],
        )

    async def act(self, messages: list[AnyMessage]) -> AIMessage:
        self.actors += 1
        if self.actors == 1:
            name, args = "read_local_file", {"path": "etl.log"}
        elif self.actors == 2:
            name, args = "calculator", {"expression": "1+1"}
        else:
            # The log exchange has left the context. These inputs exist only in notes.
            assert self.actors == 3
            assert not any(
                isinstance(message, ToolMessage) and message.name == "read_local_file"
                for message in messages
            )
            memory = WorkingMemory.model_validate_json(str(messages[2].content).split("\n")[1])
            counts = dict(pair.split("=") for pair in memory.notes[0].split())
            name, args = (
                "calculator",
                {"expression": f"{counts['rejected']}/{counts['expected']}*100"},
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"call_{self.actors}",
                }
            ],
        )

    async def update_memory(self, context: MemoryInput) -> WorkingMemory:
        self.updates.append(context)
        latest = context.results[-1]
        notes = context.notes
        if latest.name == "read_local_file":
            content = json.loads(str(latest.message.content))["content"]
            notes = [content]
        criterion = DoneCriterion(description=context.done_criteria[0].description)
        if latest.name == "calculator" and json.loads(str(latest.message.content))["result"] == 25:
            criterion = DoneCriterion(
                description=criterion.description, satisfied=True, evidence=[latest.id]
            )
        return WorkingMemory(
            observations=[Observation(summary=str(latest.message.content), evidence=[latest.id])],
            notes=notes,
            done_criteria=[criterion],
        )

    async def finalize(self, messages: list[AnyMessage]) -> FinalResponse:
        self.final_messages = messages
        return FinalResponse(
            final_answer="Отклонено 25%",
            next_action=NextAction(
                action="Проверить отклонённые строки", rationale="25% отклонений"
            ),
        )


async def test_notes_drive_later_action_and_criteria_stop_graph(tmp_path: Path) -> None:
    (tmp_path / "etl.log").write_text("expected=120 rejected=30")
    planner = MemoryPlanner()
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal="Прочитай лог и вычисли процент")
        )
    )
    assert planner.actors == 3 and len(planner.updates) == 3
    assert state.stop_reason == "criteria_met"
    assert state.done_criteria[0].evidence == ["result_3"]
    assert "25%" in str(state.final_answer)
    assert len(state.tool_results) == 3
    assert any(
        isinstance(item, ToolMessage) and item.name == "read_local_file" for item in state.messages
    )
    assert not any(
        isinstance(item, ToolMessage) and item.name == "read_local_file"
        for item in planner.final_messages
    )
    deltas = [entry for entry in state.history if entry.state_delta is not None]
    assert [entry.node for entry in deltas] == [
        "input",
        "planning",
        "acting",
        "tools",
        "observation",
        "acting",
        "tools",
        "observation",
        "acting",
        "tools",
        "observation",
        "finalization",
    ]
    assert all(
        "messages" not in entry.state_delta and "history" not in entry.state_delta
        for entry in deltas
        if entry.state_delta
    )
    assert AgentState.model_validate_json(state.model_dump_json()) == state


async def test_repeated_batch_uses_saved_success_and_error_and_stops(tmp_path: Path) -> None:
    # Missing file produces a cached error; calculator produces a cached success.
    state = AgentState.model_validate(
        await build_graph(ToolPlanner(repeat=True), tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal="Inspect", max_steps=5)
        )
    )
    assert state.current_step == 2 and state.stop_reason == "no_new_results"
    assert len(state.tool_results) == 2
    results = [item for item in state.messages if isinstance(item, ToolMessage)]
    assert [item.status for item in results] == ["success", "error", "success", "error"]
    assert [item.tool_call_id for item in results] == ["calc_1", "read_1", "calc_2", "read_2"]
    assert all(item.additional_kwargs["reused"] for item in results[-2:])
    assert "новых данных нет" in str(state.final_answer)


async def test_duplicate_calls_in_batch_execute_once_with_canonical_args() -> None:
    executions = 0

    @tool
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        nonlocal executions
        executions += 1
        return str(a + b)

    class DuplicatePlanner(FakePlanner):
        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"a": 1, "b": 2}, "id": "first"},
                    {"name": "add", "args": {"b": 2, "a": 1}, "id": "second"},
                ],
            )

    state = AgentState.model_validate(
        await build_graph(DuplicatePlanner(), tools=[add]).ainvoke(
            RunInput(goal="Add", max_steps=1)
        )
    )
    assert executions == 1 and len(state.tool_results) == 1
    results = [item for item in state.messages if isinstance(item, ToolMessage)]
    assert [item.tool_call_id for item in results] == ["first", "second"]
    assert results[-1].additional_kwargs["reused"] is True


@pytest.mark.parametrize("boundary", ["tools", "observation"])
async def test_resume_preserves_memory_and_cache(boundary: str, tmp_path: Path) -> None:
    (tmp_path / "etl.log").write_text("expected=120 rejected=30")
    planner = MemoryPlanner()
    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(planner, tools=create_tools(tmp_path), checkpointer=saver)
    await graph.ainvoke(RunInput(goal="Inspect"), config, interrupt_after=[boundary])
    snapshot = await graph.aget_state(config)
    assert len(snapshot.values["tool_results"]) == 1
    if boundary == "observation":
        assert snapshot.values["notes"] == ["expected=120 rejected=30"]
    state = AgentState.model_validate(
        await build_graph(planner, tools=create_tools(tmp_path), checkpointer=saver).ainvoke(
            None, config
        )
    )
    assert state.stop_reason == "criteria_met"
    assert len(planner.updates) == 3 and len(state.tool_results) == 3


async def test_observation_retry_does_not_repeat_tools_or_history(tmp_path: Path) -> None:
    (tmp_path / "etl.log").write_text("expected=120 rejected=30")

    class RetryingMemory(MemoryPlanner):
        attempts = 0

        async def update_memory(self, context: MemoryInput) -> WorkingMemory:
            self.attempts += 1
            if self.attempts == 1:
                raise TransientPlanningError("busy")
            return await super().update_memory(context)

    planner = RetryingMemory()
    state = AgentState.model_validate(
        await build_graph(
            planner, tools=create_tools(tmp_path), retry_policy=fast_retries()
        ).ainvoke(RunInput(goal="Inspect"))
    )
    assert planner.attempts == 4 and len(state.tool_results) == 3
    assert len([item for item in state.history if item.node == "tools"]) == 3
    assert len([item for item in state.history if item.node == "observation"]) == 3


@pytest.mark.parametrize("invalid", ["unknown", "empty", "changed"])
async def test_invalid_criterion_evidence_is_rejected(invalid: str, tmp_path: Path) -> None:
    class InvalidMemory(FakePlanner):
        async def create_plan(self, goal: str) -> ExecutionPlan:
            return ExecutionPlan(
                steps=["Read"], done_criteria=[DoneCriterion(description="Read file")]
            )

        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_local_file",
                        "args": {"path": "missing.log"},
                        "id": "read",
                    }
                ],
            )

        async def update_memory(self, context: MemoryInput) -> WorkingMemory:
            evidence = {"unknown": ["unknown"], "failed": ["result_1"], "empty": [], "changed": []}[
                invalid
            ]
            return WorkingMemory(
                done_criteria=[
                    DoneCriterion(
                        description="Different" if invalid == "changed" else "Read file",
                        satisfied=True,
                        evidence=evidence,
                    )
                ]
            )

    with pytest.raises(PlanningError):
        await build_graph(InvalidMemory(), tools=create_tools(tmp_path)).ainvoke(
            RunInput(goal="Read")
        )


@pytest.mark.parametrize(
    "field,count,size",
    [
        ("notes", 9, 1),
        ("notes", 1, 401),
        ("observations", 11, 1),
        ("observations", 1, 501),
        ("done_criteria", 6, 1),
        ("done_criteria", 1, 301),
    ],
)
def test_memory_limits_are_enforced(field: str, count: int, size: int) -> None:
    item: object = "x" * size
    if field == "observations":
        item = {"summary": item, "evidence": ["result_1"]}
    elif field == "done_criteria":
        item = {"description": item}
    with pytest.raises(ValidationError):
        WorkingMemory.model_validate({field: [item] * count})


def test_old_result_has_empty_working_memory() -> None:
    state = AgentState.model_validate({"goal": "Old", "final_answer": "Answer"})
    assert (
        not state.observations
        and not state.tool_results
        and not state.notes
        and not state.done_criteria
    )
    assert state.stop_reason is None


@pytest.mark.parametrize("boundary", ["tools", "observation"])
async def test_cache_survives_resume_and_prevents_actual_reexecution(boundary: str) -> None:
    executions = 0

    @tool
    def calculator(expression: str) -> str:
        """Count real executions across graph reconstruction."""
        nonlocal executions
        executions += 1
        return '{"result":30}'

    planner = ToolPlanner(repeat=True)
    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(planner, tools=[calculator], checkpointer=saver)
    await graph.ainvoke(RunInput(goal="Inspect"), config, interrupt_after=[boundary])
    assert executions == 1
    state = AgentState.model_validate(
        await build_graph(planner, tools=[calculator], checkpointer=saver).ainvoke(None, config)
    )
    assert executions == 1 and state.stop_reason == "no_new_results"
    assert len(state.tool_results) == 2  # Successful calculator and unknown file tool error.


async def test_bad_observation_update_leaves_tools_checkpoint_intact(tmp_path: Path) -> None:
    class InvalidObservation(ToolPlanner):
        async def update_memory(self, context: MemoryInput) -> WorkingMemory:
            return WorkingMemory(
                observations=[Observation(summary="Unverified", evidence=["missing"])]
            )

    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_graph(InvalidObservation(), tools=create_tools(tmp_path), checkpointer=saver)
    with pytest.raises(PlanningError, match="unknown tool result"):
        await graph.ainvoke(RunInput(goal="Inspect", max_steps=1), config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("observation",)
    saved = AgentState.model_validate(snapshot.values)
    assert saved.observations == [] and saved.notes == []
    assert len(snapshot.values["tool_results"]) == len(snapshot.values["pending_observations"]) == 2
    state = AgentState.model_validate(
        await build_graph(FakePlanner(), tools=create_tools(tmp_path), checkpointer=saver).ainvoke(
            None, config
        )
    )
    assert state.observations and not state.pending_observations
    assert len(state.tool_results) == 2 and state.stop_reason == "step_limit"


@pytest.mark.parametrize("corrects_update", [True, False])
async def test_memory_validation_correction_is_bounded_and_does_not_repeat_tools(
    corrects_update: bool,
) -> None:
    calls = 0

    @tool
    def calculate() -> str:
        """Calculate the demo result."""
        nonlocal calls
        calls += 1
        return '{"result":4}'

    class CorrectingPlanner(FakePlanner):
        attempts = 0

        async def create_plan(self, goal: str) -> ExecutionPlan:
            return ExecutionPlan(
                steps=["Calculate"], done_criteria=[DoneCriterion(description="Calculate")]
            )

        async def act(self, messages: list[AnyMessage]) -> AIMessage:
            return AIMessage(
                content="", tool_calls=[{"name": "calculate", "args": {}, "id": "calc"}]
            )

        async def update_memory(self, context: MemoryInput) -> WorkingMemory:
            self.attempts += 1
            if self.attempts == 1:
                assert context.validation_feedback is None
            else:
                assert context.validation_feedback
                assert "missing or unsuccessful evidence" in context.validation_feedback
                assert "unknown_result" in context.validation_feedback
            return WorkingMemory(
                done_criteria=[
                    DoneCriterion(
                        description="Calculate",
                        satisfied=True,
                        evidence=[
                            "result_1"
                            if corrects_update and self.attempts == 2
                            else "unknown_result"
                        ],
                    )
                ]
            )

    planner = CorrectingPlanner()
    graph = build_graph(planner, tools=[calculate])
    if corrects_update:
        state = AgentState.model_validate(await graph.ainvoke(RunInput(goal="Calculate")))
        assert state.stop_reason == "criteria_met"
        assert state.done_criteria[0].evidence == ["result_1"]
        assert len(state.tool_results) == 1
    else:
        with pytest.raises(PlanningError, match="unsuccessful evidence"):
            await graph.ainvoke(RunInput(goal="Calculate"))
    assert planner.attempts == 2
    assert calls == 1


@pytest.mark.parametrize("satisfied", [True, False])
def test_failed_evidence_is_removed_and_criterion_remains_unconfirmed(satisfied: bool) -> None:
    from etl_agent.domain import ToolResult

    state = AgentState(
        goal="Read and calculate",
        done_criteria=[DoneCriterion(description="Read")],
        tool_results=[
            ToolResult(
                id="result_1",
                name="calculator",
                args={},
                message=ToolMessage(content="4", tool_call_id="calc", status="success"),
            ),
            ToolResult(
                id="result_2",
                name="read_local_file",
                args={},
                message=ToolMessage(content="File not found", tool_call_id="read", status="error"),
            ),
        ],
    )
    memory = WorkingMemory(
        observations=[Observation(summary="File unavailable", evidence=["result_2"])],
        done_criteria=[
            DoneCriterion(
                description="Read", satisfied=satisfied, evidence=["result_1", "result_2"]
            )
        ],
    )
    corrected = reject_failed_evidence(state, memory)
    validate_memory(state, corrected)
    assert not corrected.done_criteria[0].satisfied
    assert corrected.done_criteria[0].evidence == ["result_1"]
    assert corrected.observations == memory.observations
    assert memory.done_criteria[0].evidence == ["result_1", "result_2"]
    unknown = corrected.model_copy(
        update={"done_criteria": [DoneCriterion(description="Read", evidence=["unknown"])]}
    )
    with pytest.raises(PlanningError, match="unsuccessful evidence"):
        validate_memory(state, reject_failed_evidence(state, unknown))
