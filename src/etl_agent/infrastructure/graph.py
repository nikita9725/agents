"""Граф с native retry/timeout policies; checkpointer хранит состояние в PostgreSQL."""

import logging
from dataclasses import dataclass
from time import perf_counter

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy

from etl_agent.application import Planner
from etl_agent.domain import AgentState, HistoryEntry, NextAction, RunInput
from etl_agent.exceptions import PlanningTimeout, StepLimitExceeded, TransientPlanningError
from etl_agent.observability import node_context

logger = logging.getLogger(__name__)


def retry_planning(exc: Exception) -> bool:
    return isinstance(exc, (TransientPlanningError, PlanningTimeout, NodeTimeoutError))


PLANNING_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2,
    max_interval=5,
    jitter=True,
    retry_on=retry_planning,
)


@dataclass(slots=True)
class InputUpdate:
    history: list[HistoryEntry]


@dataclass(slots=True)
class PlanningUpdate:
    next_action: NextAction
    current_step: int
    history: list[HistoryEntry]


@dataclass(slots=True)
class FinalizationUpdate:
    final_answer: str
    history: list[HistoryEntry]


class Nodes:
    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    @staticmethod
    async def initialize(state: RunInput) -> InputUpdate:
        return InputUpdate(history=[HistoryEntry(role="user", node="input", content=state.goal)])

    async def planning(
        self,
        state: AgentState,
        runtime: Runtime[None],
    ) -> PlanningUpdate:
        if state.current_step >= state.max_steps:
            raise StepLimitExceeded("Planning step budget exhausted")
        with node_context(get_config(), runtime, state.goal):
            started = perf_counter()
            info = runtime.execution_info
            assert info is not None
            attempt = info.node_attempt
            logger.info("event=node.started node=planning attempt=%s", attempt)
            try:
                action = await self._planner.propose(state.goal)
            except Exception:
                logger.exception("event=node.failed node=planning attempt=%s", attempt)
                raise
            logger.info(
                "event=node.completed node=planning attempt=%s elapsed_ms=%.1f",
                attempt,
                (perf_counter() - started) * 1000,
            )
            return PlanningUpdate(
                next_action=action,
                current_step=state.current_step + 1,
                history=[
                    *state.history,
                    HistoryEntry(
                        role="assistant",
                        node="planning",
                        content=f"{action.action}\n{action.rationale}",
                    ),
                ],
            )

    @staticmethod
    async def finalization(
        state: AgentState,
        runtime: Runtime[None],
    ) -> FinalizationUpdate:
        with node_context(get_config(), runtime, state.goal):
            if state.next_action is None:
                raise ValueError("Cannot finalize without a proposed action")
            answer = f"Следующий шаг: {state.next_action.action}\n{state.next_action.rationale}"
            logger.info(
                "event=node.completed node=finalization attempt=%s",
                runtime.execution_info.node_attempt if runtime.execution_info else 1,
            )
            return FinalizationUpdate(
                final_answer=answer,
                history=[
                    *state.history,
                    HistoryEntry(
                        role="assistant",
                        node="finalization",
                        content=answer,
                    ),
                ],
            )


def build_graph[Version: (int, float, str)](
    planner: Planner,
    *,
    checkpointer: BaseCheckpointSaver[Version] | None = None,
    retry_policy: RetryPolicy = PLANNING_RETRY,
    planning_timeout: float = 30,
) -> CompiledStateGraph[AgentState, None, RunInput, AgentState]:
    nodes = Nodes(planner)
    builder = StateGraph(AgentState, input_schema=RunInput)
    builder.add_node("input", nodes.initialize, timeout=5)
    builder.add_node(
        "planning",
        nodes.planning,
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_node("finalization", nodes.finalization, timeout=5)
    builder.add_edge(START, "input")
    builder.add_edge("input", "planning")
    builder.add_edge("planning", "finalization")
    builder.add_edge("finalization", END)
    return builder.compile(checkpointer=checkpointer)
