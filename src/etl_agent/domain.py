"""Модели состояния и результатов, независимые от внешних фреймворков."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NextAction(Model):
    action: NonEmptyText = Field(description="One concrete next diagnostic action")
    rationale: NonEmptyText = Field(description="Brief justification based on the user's input")


class HistoryEntry(Model):
    role: Literal["user", "assistant"]
    node: Literal["input", "planning", "finalization"]
    content: NonEmptyText


class RunInput(Model):
    goal: NonEmptyText
    max_steps: int = Field(default=1, gt=0)


class AgentState(RunInput):
    history: list[HistoryEntry] = Field(default_factory=list)
    current_step: int = Field(default=0, ge=0)
    next_action: NextAction | None = None
    final_answer: NonEmptyText | None = None
