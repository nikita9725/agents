"""Порт планировщика без зависимости от runtime."""

from typing import Protocol

from langchain_core.messages import AIMessage, AnyMessage

from etl_agent.domain import ExecutionPlan, FinalResponse, MemoryInput, NextAction, WorkingMemory


class Planner(Protocol):
    async def propose(self, goal: str) -> NextAction: ...

    async def create_plan(self, goal: str) -> ExecutionPlan: ...

    async def act(self, messages: list[AnyMessage]) -> AIMessage: ...

    async def finalize(self, messages: list[AnyMessage]) -> FinalResponse: ...

    async def update_memory(self, context: MemoryInput) -> WorkingMemory: ...
