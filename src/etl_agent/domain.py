"""Типизированные модели состояния графа, сообщений и результатов."""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NextAction(Model):
    action: NonEmptyText = Field(description="One concrete next diagnostic action")
    rationale: NonEmptyText = Field(description="Brief justification based on the user's input")


NodeName = Literal["input", "planning", "acting", "tools", "observation", "finalization"]


class ExecutionPlan(Model):
    steps: list[NonEmptyText] = Field(
        min_length=1,
        description="Ordered brief steps, each item is a plain string, never an object",
    )


class HistoryEntry(Model):
    role: Literal["user", "assistant", "tool"]
    node: NodeName
    content: NonEmptyText


class RunInput(Model):
    goal: NonEmptyText
    max_steps: int = Field(default=5, gt=0)


class AgentState(RunInput):
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
