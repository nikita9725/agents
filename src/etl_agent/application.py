"""Порт планировщика без зависимости от runtime."""

from typing import Protocol

from langchain_core.messages import AIMessage, AnyMessage

from etl_agent.domain import FinalResponse, NextAction


class Planner(Protocol):
    async def propose(self, goal: str) -> NextAction: ...

    async def plan(self, messages: list[AnyMessage]) -> AIMessage: ...

    async def finalize(self, messages: list[AnyMessage]) -> FinalResponse: ...
