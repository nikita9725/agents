"""Порт планировщика без зависимости от runtime."""

from typing import Protocol

from etl_agent.domain import NextAction


class Planner(Protocol):
    async def propose(self, goal: str) -> NextAction: ...
