"""Durable tool-calling graph with bounded planning and persisted observations."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy

from etl_agent.application import Planner
from etl_agent.domain import AgentState, ExecutionPlan, HistoryEntry, NextAction, RunInput
from etl_agent.exceptions import PlanningTimeout, StepLimitExceeded, TransientPlanningError
from etl_agent.infrastructure.tools import create_tools
from etl_agent.observability import node_context, tool_log_payload

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
    messages: list[AnyMessage]
    history: list[HistoryEntry]


@dataclass(slots=True)
class PlanningUpdate:
    plan: ExecutionPlan
    messages: list[AnyMessage]
    history: list[HistoryEntry]


@dataclass(slots=True)
class ActingUpdate:
    messages: list[AnyMessage]
    current_step: int
    history: list[HistoryEntry]


@dataclass(slots=True)
class ToolsUpdate:
    pending_observations: list[ToolMessage]


@dataclass(slots=True)
class ObservationUpdate:
    messages: list[AnyMessage]
    history: list[HistoryEntry]
    pending_observations: list[ToolMessage]


@dataclass(slots=True)
class FinalizationUpdate:
    next_action: NextAction
    final_answer: str
    history: list[HistoryEntry]


class Nodes:
    def __init__(self, planner: Planner, tools: list[BaseTool]) -> None:
        self._planner = planner
        # ToolNode's default handles invocation errors but propagates unexpected tool bugs.
        self._tools = ToolNode(tools)

    @staticmethod
    async def initialize(state: RunInput) -> InputUpdate:
        return InputUpdate(
            messages=[HumanMessage(content=state.goal)],
            history=[HistoryEntry(role="user", node="input", content=state.goal)],
        )

    async def planning(self, state: AgentState, runtime: Runtime[None]) -> PlanningUpdate:
        with node_context(get_config(), runtime, state.goal):
            started = perf_counter()
            attempt = runtime.execution_info.node_attempt if runtime.execution_info else 1
            logger.info("event=node.started node=planning attempt=%s", attempt)
            try:
                plan = await self._planner.create_plan(state.goal)
            except Exception:
                logger.exception("event=node.failed node=planning attempt=%s", attempt)
                raise
            text = "План:\n" + "\n".join(
                f"{index}. {step}" for index, step in enumerate(plan.steps, 1)
            )
            logger.info(
                "event=node.completed node=planning attempt=%s elapsed_ms=%.1f",
                attempt,
                (perf_counter() - started) * 1000,
            )
            return PlanningUpdate(
                plan=plan,
                messages=[*state.messages, AIMessage(content=text)],
                history=[
                    *state.history,
                    HistoryEntry(role="assistant", node="planning", content=text),
                ],
            )

    async def acting(self, state: AgentState, runtime: Runtime[None]) -> ActingUpdate:
        if state.current_step >= state.max_steps:
            raise StepLimitExceeded("Action step budget exhausted")
        with node_context(get_config(), runtime, state.goal):
            started = perf_counter()
            attempt = runtime.execution_info.node_attempt if runtime.execution_info else 1
            logger.info("event=node.started node=acting attempt=%s", attempt)
            try:
                message = await self._planner.act(state.messages)
            except Exception:
                logger.exception("event=node.failed node=acting attempt=%s", attempt)
                raise
            logger.info(
                "event=node.completed node=acting attempt=%s elapsed_ms=%.1f",
                attempt,
                (perf_counter() - started) * 1000,
            )
            parts: list[str] = [message.text] if message.text else []
            if message.tool_calls:
                parts.append(json.dumps(message.tool_calls, ensure_ascii=False))
            return ActingUpdate(
                messages=[*state.messages, message],
                current_step=state.current_step + 1,
                history=[
                    *state.history,
                    HistoryEntry(
                        role="assistant",
                        node="acting",
                        content="\n".join(parts) or "Диагностика завершена",
                    ),
                ],
            )

    async def tools(self, state: AgentState, runtime: Runtime[None]) -> ToolsUpdate:
        with node_context(get_config(), runtime, state.goal):
            message = state.messages[-1]
            if not isinstance(message, AIMessage):
                raise ValueError("Tools require an AIMessage")
            # Execute individually so logs pair each call with its own result and duration.
            results: list[ToolMessage] = []
            for call in message.tool_calls:
                started = perf_counter()
                logger.info(
                    "event=tool.started name=%s tool_call_id=%s args=%s",
                    call["name"],
                    call["id"],
                    tool_log_payload(call["args"]),
                )
                try:
                    output = await self._tools.ainvoke([call], get_config())
                except Exception:
                    logger.exception(
                        "event=tool.failed name=%s tool_call_id=%s", call["name"], call["id"]
                    )
                    raise
                result = output["messages"][0]
                if not isinstance(result, ToolMessage):
                    raise TypeError("ToolNode must return ToolMessage")
                logger.info(
                    "event=tool.completed name=%s tool_call_id=%s status=%s elapsed_ms=%.1f result=%s",
                    call["name"],
                    call["id"],
                    result.status,
                    (perf_counter() - started) * 1000,
                    tool_log_payload(result.content),
                )
                results.append(result)
            return ToolsUpdate(pending_observations=results)

    @staticmethod
    async def observation(state: AgentState) -> ObservationUpdate:
        return ObservationUpdate(
            messages=[*state.messages, *state.pending_observations],
            history=[
                *state.history,
                *[
                    HistoryEntry(role="tool", node="observation", content=result.model_dump_json())
                    for result in state.pending_observations
                ],
            ],
            pending_observations=[],
        )

    @staticmethod
    def after_acting(state: AgentState) -> Literal["tools", "finalization"]:
        last = state.messages[-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalization"

    @staticmethod
    def after_observation(state: AgentState) -> Literal["acting", "finalization"]:
        return "acting" if state.current_step < state.max_steps else "finalization"

    async def finalization(self, state: AgentState, runtime: Runtime[None]) -> FinalizationUpdate:
        with node_context(get_config(), runtime, state.goal):
            messages: list[AnyMessage] = list(state.messages)
            exhausted = state.current_step >= state.max_steps and isinstance(
                messages[-1], ToolMessage
            )
            if exhausted:
                messages.append(
                    HumanMessage(
                        content="Бюджет инструментальных шагов исчерпан. Заверши по имеющимся наблюдениям и укажи это ограничение."
                    )
                )
            response = await self._planner.finalize(messages)
            answer = response.final_answer
            if exhausted:
                answer += "\nБюджет инструментальных шагов исчерпан; вывод основан на полученных наблюдениях."
            logger.info(
                "event=node.completed node=finalization attempt=%s",
                runtime.execution_info.node_attempt if runtime.execution_info else 1,
            )
            return FinalizationUpdate(
                next_action=response.next_action,
                final_answer=answer,
                history=[
                    *state.history,
                    HistoryEntry(role="assistant", node="finalization", content=answer),
                ],
            )


def build_graph[Version: (int, float, str)](
    planner: Planner,
    *,
    tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver[Version] | None = None,
    retry_policy: RetryPolicy = PLANNING_RETRY,
    planning_timeout: float = 30,
) -> CompiledStateGraph[AgentState, None, RunInput, AgentState]:
    nodes = Nodes(planner, tools if tools is not None else create_tools(Path("examples/tool_data")))
    builder = StateGraph(AgentState, input_schema=RunInput)
    builder.add_node("input", nodes.initialize, timeout=5)
    builder.add_node(
        "planning",
        nodes.planning,
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_node(
        "acting",
        nodes.acting,
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_node("tools", nodes.tools, timeout=10)
    builder.add_node("observation", nodes.observation, timeout=5)
    builder.add_node(
        "finalization",
        nodes.finalization,
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_edge(START, "input")
    builder.add_edge("input", "planning")
    builder.add_edge("planning", "acting")
    builder.add_conditional_edges("acting", nodes.after_acting)
    builder.add_edge("tools", "observation")
    builder.add_conditional_edges("observation", nodes.after_observation)
    builder.add_edge("finalization", END)
    return builder.compile(checkpointer=checkpointer)
