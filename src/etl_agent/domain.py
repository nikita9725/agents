"""Типизированные модели состояния графа, сообщений и результатов."""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NextAction(Model):
    action: NonEmptyText = Field(description="One concrete next diagnostic action")
    rationale: NonEmptyText = Field(description="Brief justification based on the user's input")


NodeName = Literal["input", "planning", "acting", "tools", "observation", "finalization"]


ShortCriterion = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)
]
ShortObservation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]
ShortNote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)]


class DoneCriterion(Model):
    description: ShortCriterion
    satisfied: bool = False
    evidence: list[NonEmptyText] = Field(default_factory=list, max_length=10)


class Observation(Model):
    summary: ShortObservation
    evidence: list[NonEmptyText] = Field(min_length=1, max_length=10)


class WorkingMemory(Model):
    observations: list[Observation] = Field(default_factory=list, max_length=10)
    notes: list[ShortNote] = Field(default_factory=list, max_length=8)
    done_criteria: list[DoneCriterion] = Field(default_factory=list, max_length=5)


class ToolResult(Model):
    id: NonEmptyText
    name: NonEmptyText
    args: dict[str, JsonValue]
    message: ToolMessage


class ExecutionPlan(Model):
    steps: list[NonEmptyText] = Field(
        min_length=1,
        description="Ordered brief steps, each item is a plain string, never an object",
    )

    done_criteria: list[DoneCriterion] = Field(default_factory=list, max_length=5)


class HistoryEntry(Model):
    role: Literal["user", "assistant", "tool"]
    node: NodeName
    content: NonEmptyText
    state_delta: dict[str, JsonValue] | None = None


class RunInput(Model):
    goal: NonEmptyText
    max_steps: int = Field(default=5, gt=0)


StopReason = Literal[
    "criteria_met", "no_new_results", "no_progress", "step_limit", "actor_finished"
]


class AgentState(RunInput, WorkingMemory):
    tool_results: list[ToolResult] = Field(default_factory=list)
    last_batch_new_results: int = Field(default=0, ge=0)
    consecutive_no_progress: int = Field(default=0, ge=0)
    stop_reason: StopReason | None = None
    plan: ExecutionPlan | None = None
    pending_observations: list[ToolMessage] = Field(default_factory=list)
    messages: list[AnyMessage] = Field(default_factory=list)
    history: list[HistoryEntry] = Field(default_factory=list)
    current_step: int = Field(default=0, ge=0)
    next_action: NextAction | None = None
    final_answer: NonEmptyText | None = None


class FinalResponse(Model):
    final_answer: NonEmptyText
    next_action: NextAction


class MemoryInput(WorkingMemory):
    goal: NonEmptyText
    plan: ExecutionPlan | None
    results: list[ToolResult]
