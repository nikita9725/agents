"""Восстанавливаемый граф вызова инструментов с ограничением шагов и сохранением наблюдений."""

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
from langgraph.errors import GraphBubbleUp, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy

from etl_agent.application import Planner
from etl_agent.domain import (
    AgentState,
    DoneCriterion,
    ExecutionPlan,
    HistoryEntry,
    NextAction,
    Observation,
    RunInput,
    StopReason,
    ToolResult,
)
from etl_agent.exceptions import PlanningError, PlanningTimeout, TransientPlanningError
from etl_agent.infrastructure.tools import ToolError, create_tools
from etl_agent.memory import call_key, memory_input, model_context, validate_memory
from etl_agent.observability import node_context, observed_node, tool_log_payload

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
    done_criteria: list[DoneCriterion]
    plan: ExecutionPlan
    messages: list[AnyMessage]
    history: list[HistoryEntry]


@dataclass(slots=True)
class ActingUpdate:
    messages: list[AnyMessage]
    current_step: int
    history: list[HistoryEntry]
    stop_reason: StopReason | None = None


@dataclass(slots=True)
class ToolsUpdate:
    pending_observations: list[ToolMessage]
    tool_results: list[ToolResult]
    last_batch_new_results: int
    history: list[HistoryEntry]


@dataclass(slots=True)
class ObservationUpdate:
    observations: list[Observation]
    notes: list[str]
    done_criteria: list[DoneCriterion]
    stop_reason: StopReason | None
    consecutive_no_progress: int
    messages: list[AnyMessage]
    history: list[HistoryEntry]
    pending_observations: list[ToolMessage]


@dataclass(slots=True)
class FinalizationUpdate:
    stop_reason: StopReason
    next_action: NextAction
    final_answer: str
    history: list[HistoryEntry]


class Nodes:
    def __init__(self, planner: Planner, tools: list[BaseTool]) -> None:
        self._planner = planner
        # Handle invocation failures at the per-call boundary, without exposing internals.
        self._tools = ToolNode(tools, handle_tool_errors=False)

    @staticmethod
    async def initialize(state: RunInput, runtime: Runtime[None]) -> InputUpdate:
        return InputUpdate(
            messages=[HumanMessage(content=state.goal)],
            history=[
                HistoryEntry(
                    role="user",
                    node="input",
                    content=state.goal,
                    state_delta={"messages_count": 1, "current_step": 0},
                )
            ],
        )

    async def planning(self, state: AgentState, runtime: Runtime[None]) -> PlanningUpdate:
        plan = await self._planner.create_plan(state.goal)
        # Planning cannot claim evidence before any tool has executed.
        criteria = [DoneCriterion(description=item.description) for item in plan.done_criteria]
        plan = plan.model_copy(update={"done_criteria": criteria})
        text = "План:\n" + "\n".join(f"{index}. {step}" for index, step in enumerate(plan.steps, 1))
        return PlanningUpdate(
            done_criteria=criteria,
            plan=plan,
            messages=[*state.messages, AIMessage(content=text)],
            history=[
                *state.history,
                HistoryEntry(
                    role="assistant",
                    node="planning",
                    content=text,
                    state_delta={
                        "plan": plan.model_dump(mode="json"),
                        "done_criteria": [item.model_dump(mode="json") for item in criteria],
                        "messages_count": len(state.messages) + 1,
                    },
                ),
            ],
        )

    async def acting(self, state: AgentState, runtime: Runtime[None]) -> ActingUpdate:
        if state.current_step >= state.max_steps:
            logger.info("event=action.chosen kind=finish stop_reason=step_limit")
            return ActingUpdate(
                messages=state.messages,
                current_step=state.current_step,
                history=[
                    *state.history,
                    HistoryEntry(
                        role="assistant",
                        node="acting",
                        content="Бюджет шагов исчерпан; переход к финализации.",
                        state_delta={"stop_reason": "step_limit"},
                    ),
                ],
                stop_reason="step_limit",
            )
        message = await self._planner.act(model_context(state))
        logger.info(
            "event=action.chosen kind=%s tool_calls=%s",
            "tools" if message.tool_calls else "finish",
            tool_log_payload(message.tool_calls),
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
                    state_delta={
                        "current_step": state.current_step + 1,
                        "messages_count": len(state.messages) + 1,
                    },
                ),
            ],
        )

    async def tools(self, state: AgentState, runtime: Runtime[None]) -> ToolsUpdate:
        message = state.messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Tools require an AIMessage")
        # Execute individually so logs pair each call with its own result and duration.
        results: list[ToolMessage] = []
        cache = {call_key(item.name, item.args): item for item in state.tool_results}
        stored = list(state.tool_results)
        reused: list[str] = []
        new_ids: list[str] = []
        for call in message.tool_calls:
            key = call_key(call["name"], call["args"])
            if key in cache:
                cached = cache[key]
                results.append(
                    cached.message.model_copy(
                        update={
                            "tool_call_id": call["id"],
                            "additional_kwargs": {"result_id": cached.id, "reused": True},
                        }
                    )
                )
                reused.append(cached.id)
                logger.info(
                    "event=tool.reused name=%s result_id=%s tool_call_id=%s",
                    call["name"],
                    cached.id,
                    call["id"],
                )
                continue
            started = perf_counter()
            result_id = f"result_{len(stored) + 1}"
            logger.info(
                "event=tool.started name=%s tool_call_id=%s result_id=%s args=%s",
                call["name"],
                call["id"],
                result_id,
                tool_log_payload(call["args"]),
            )
            try:
                output = await self._tools.ainvoke([call], get_config())
            except GraphBubbleUp:
                # Native graph interrupts and control flow are not tool failures.
                raise
            except Exception:
                logger.exception(
                    "event=tool.failed name=%s tool_call_id=%s", call["name"], call["id"]
                )
                result = ToolMessage(
                    content=ToolError(
                        error="Tool execution failed; check arguments or choose another tool"
                    ).model_dump_json(),
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            else:
                result = output["messages"][0]
            if not isinstance(result, ToolMessage):
                raise TypeError("ToolNode must return ToolMessage")
            logger.info(
                "event=tool.completed name=%s tool_call_id=%s status=%s result_id=%s "
                "elapsed_ms=%.1f result=%s",
                call["name"],
                call["id"],
                result.status,
                result_id,
                (perf_counter() - started) * 1000,
                tool_log_payload(result.content),
            )
            result = result.model_copy(
                update={
                    "additional_kwargs": {
                        **result.additional_kwargs,
                        "result_id": result_id,
                        "reused": False,
                    },
                }
            )
            item = ToolResult(id=result_id, name=call["name"], args=call["args"], message=result)
            logger.info(
                "event=observation.received name=%s tool_call_id=%s result_id=%s status=%s",
                call["name"],
                call["id"],
                result_id,
                result.status,
            )
            stored.append(item)
            cache[key] = item
            new_ids.append(result_id)
            results.append(result)
        new_successes = sum(
            item.message.status == "success" for item in stored if item.id in new_ids
        )
        return ToolsUpdate(
            pending_observations=results,
            tool_results=stored,
            last_batch_new_results=len(new_ids),
            history=[
                *state.history,
                HistoryEntry(
                    role="assistant",
                    node="tools",
                    content=f"Новых результатов: {len(new_ids)}; из памяти: {len(reused)}",
                    state_delta={
                        "new_result_ids": new_ids,
                        "reused_result_ids": reused,
                        "pending_observations": len(results),
                        "last_batch_new_results": len(new_ids),
                        "new_successful_results": new_successes,
                    },
                ),
            ],
        )

    async def observation(self, state: AgentState, runtime: Runtime[None]) -> ObservationUpdate:
        context = memory_input(state)
        memory = await self._planner.update_memory(context)
        try:
            validate_memory(state, memory)
        except PlanningError as error:
            # One bounded correction; never repeat tools or accept invalid evidence.
            logger.warning("event=memory.validation_failed correction_attempt=1")
            context = context.model_copy(
                update={
                    "validation_feedback": (
                        f"Previous memory update was rejected: {error}. "
                        "Correct the rejected update using only the original results and criteria. "
                        "Rejected update: " + memory.model_dump_json()
                    )
                }
            )
            memory = await self._planner.update_memory(context)
            validate_memory(state, memory)
        has_progress = any(
            result.status == "success" and not result.additional_kwargs.get("reused", False)
            for result in state.pending_observations
        )
        consecutive_no_progress = 0 if has_progress else state.consecutive_no_progress + 1
        reason: StopReason | None = None
        if memory.done_criteria and all(item.satisfied for item in memory.done_criteria):
            reason = "criteria_met"
        elif state.last_batch_new_results == 0:
            reason = "no_new_results"
        elif consecutive_no_progress >= 2:
            reason = "no_progress"
        elif state.current_step >= state.max_steps:
            reason = "step_limit"
        entries = [
            HistoryEntry(role="tool", node="observation", content=result.model_dump_json())
            for result in state.pending_observations
        ]
        # One durable delta per graph node, attached to its final history entry.
        delta = memory.model_dump(mode="json")
        delta.update(
            {
                "pending_observations": 0,
                "stop_reason": reason,
                "consecutive_no_progress": consecutive_no_progress,
                "messages_count": len(state.messages) + len(entries),
            }
        )
        entries[-1] = entries[-1].model_copy(update={"state_delta": delta})
        return ObservationUpdate(
            observations=memory.observations,
            notes=memory.notes,
            done_criteria=memory.done_criteria,
            stop_reason=reason,
            consecutive_no_progress=consecutive_no_progress,
            messages=[*state.messages, *state.pending_observations],
            history=[*state.history, *entries],
            pending_observations=[],
        )

    @staticmethod
    def after_acting(state: AgentState, runtime: Runtime[None]) -> Literal["tools", "finalization"]:
        last = state.messages[-1]
        chosen: Literal["tools", "finalization"] = (
            "finalization"
            if state.stop_reason
            else "tools"
            if isinstance(last, AIMessage) and last.tool_calls
            else "finalization"
        )
        with node_context(get_config(), runtime, state.goal):
            logger.info(
                "event=node.chosen node=acting next_node=%s stop_reason=%s",
                chosen,
                state.stop_reason or ("actor_finished" if chosen == "finalization" else None),
            )
        return chosen

    @staticmethod
    def after_observation(
        state: AgentState, runtime: Runtime[None]
    ) -> Literal["acting", "finalization"]:
        chosen: Literal["acting", "finalization"] = (
            "finalization" if state.stop_reason else "acting"
        )
        with node_context(get_config(), runtime, state.goal):
            logger.info(
                "event=node.chosen node=observation next_node=%s stop_reason=%s",
                chosen,
                state.stop_reason,
            )
        return chosen

    async def finalization(self, state: AgentState, runtime: Runtime[None]) -> FinalizationUpdate:
        messages: list[AnyMessage] = model_context(state)
        reason = state.stop_reason or "actor_finished"
        messages.append(HumanMessage(content=f"Причина завершения диагностики: {reason}."))
        exhausted = reason == "step_limit"
        if exhausted:
            messages.append(
                HumanMessage(
                    content="Бюджет инструментальных шагов исчерпан. Заверши по имеющимся наблюдениям и укажи это ограничение."
                )
            )
        response = await self._planner.finalize(messages)
        answer = response.final_answer
        if exhausted:
            answer += (
                "\nБюджет инструментальных шагов исчерпан; вывод основан на полученных наблюдениях."
            )
        if reason == "criteria_met":
            answer += "\nКритерии диагностики выполнены по сохранённым наблюдениям."
        elif reason == "no_new_results":
            answer += "\nПовторные вызовы использовали сохранённые результаты; новых данных нет."
        elif reason == "no_progress":
            answer += (
                "\nДиагностика остановлена: два пакета подряд не дали новых успешных результатов."
            )
        remaining = [item.description for item in state.done_criteria if not item.satisfied]
        if remaining:
            answer += "\nНе подтверждено: " + "; ".join(remaining)
        logger.info(
            "event=diagnosis.completed stop_reason=%s current_step=%s max_steps=%s",
            reason,
            state.current_step,
            state.max_steps,
        )
        return FinalizationUpdate(
            stop_reason=reason,
            next_action=response.next_action,
            final_answer=answer,
            history=[
                *state.history,
                HistoryEntry(
                    role="assistant",
                    node="finalization",
                    content=answer,
                    state_delta={
                        "stop_reason": reason,
                        "final_answer": answer,
                        "next_action": response.next_action.model_dump(mode="json"),
                    },
                ),
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
    builder.add_node(
        "input", observed_node("input", nodes.initialize), input_schema=RunInput, timeout=5
    )
    builder.add_node(
        "planning",
        observed_node("planning", nodes.planning),
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_node(
        "acting",
        observed_node("acting", nodes.acting),
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_node("tools", observed_node("tools", nodes.tools), timeout=10)
    builder.add_node(
        "observation",
        observed_node("observation", nodes.observation),
        retry_policy=retry_policy,
        timeout=TimeoutPolicy(run_timeout=planning_timeout),
    )
    builder.add_node(
        "finalization",
        observed_node("finalization", nodes.finalization),
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
