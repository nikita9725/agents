"""Контекст запроса и безопасный вывод диагностических сообщений."""

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from time import perf_counter
from types import TracebackType
from typing import Protocol
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langsmith import Client, tracing_context
from pydantic import ValidationError
from requests import Session

from etl_agent.domain import HistoryEntry, RunInput
from etl_agent.settings import Settings

logger = logging.getLogger(__name__)

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)
task_id: ContextVar[str | None] = ContextVar("task_id", default=None)
redacted_secrets: set[str] = set()
sensitive_values: ContextVar[tuple[str, ...]] = ContextVar("sensitive_values", default=())


def protect_text(*values: str) -> None:
    sensitive_values.set((*sensitive_values.get(), *(value for value in values if value)))


def tool_log_payload(value: object) -> str:
    """Redact before truncation so a boundary cannot expose part of a known secret."""
    text = json.dumps(value, ensure_ascii=False)
    for secret in sorted(redacted_secrets.union(sensitive_values.get()), key=len, reverse=True):
        if secret:
            text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
    return text[:1024]


class DiagnosticFormatter(logging.Formatter):
    def formatException(
        self,
        ei: tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None],
    ) -> str:
        text = super().formatException(ei)
        error = ei[1]
        visited: set[int] = set()
        while error is not None and id(error) not in visited:
            visited.add(id(error))
            if isinstance(error, ValidationError):
                text = text.replace(
                    str(error),
                    str(
                        error.errors(
                            include_input=False,
                            include_context=False,
                            include_url=False,
                        )
                    ),
                )
            elif isinstance(error, httpx.HTTPError):
                # SDK status errors may embed upstream response bodies and credentials.
                text = text.replace(
                    str(error), f"{type(error).__name__} (upstream details omitted)"
                )
            elif isinstance(error, OutputParserException):
                text = text.replace(str(error), "Invalid model output (content omitted)")
            error = error.__cause__ or error.__context__
        return text

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id.get() or "-"
        record.run_id = run_id.get() or "-"
        record.thread_id = thread_id.get() or "-"
        record.task_id = task_id.get() or "-"
        message = super().format(record)
        for secret in sorted(redacted_secrets.union(sensitive_values.get()), key=len, reverse=True):
            message = message.replace(secret, "[REDACTED]")
        message = re.sub(r"postgres(?:ql)?(?:\+[a-z0-9_]+)?://[^\s\"']+", "[REDACTED_DSN]", message)
        return re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", message)


def configure_logging(
    secrets: list[str] | None = None,
    *,
    native_runtime: bool = False,
) -> None:
    values = {**dotenv_values(".env"), **os.environ}
    redacted_secrets.update(
        value
        for name in (
            "DATABASE_URL",
            "COMPOSE_DATABASE_URL",
            "TASK_DATABASE_URL",
            "LLM_API_KEY",
            "LANGSMITH_API_KEY",
        )
        if (value := values.get(name))
    )
    redacted_secrets.update(value for value in secrets or [] if value)
    root = logging.getLogger("etl_agent" if native_runtime else "")
    if native_runtime:
        root.propagate = False
    if not any(getattr(handler, "etl_diagnostics", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.etl_diagnostics = True  # type: ignore[attr-defined]
        handler.setFormatter(
            DiagnosticFormatter(
                "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s "
                "task_id=%(task_id)s thread_id=%(thread_id)s run_id=%(run_id)s %(message)s"
            )
        )
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    if native_runtime:
        return
    for name in ("uvicorn", "uvicorn.error"):
        for existing_handler in logging.getLogger(name).handlers:
            existing_handler.setFormatter(
                DiagnosticFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
    # Our middleware replaces the access log without query strings.
    logging.getLogger("uvicorn.access").disabled = True
    # HTTP SDK logs can contain provider URLs; application logs record call outcomes.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@contextmanager
def node_context(
    config: RunnableConfig,
    runtime: Runtime[None],
    goal: str,
) -> Iterator[None]:
    metadata = config.get("metadata", {})
    info = runtime.execution_info
    if info is None:
        raise RuntimeError("Node context requires active LangGraph execution info")
    tokens = (
        request_id.set(metadata.get("request_id")),
        run_id.set(info.run_id),
        thread_id.set(info.thread_id),
        sensitive_values.set((goal,)),
        task_id.set(metadata.get("task_id")),
    )
    try:
        yield
    finally:
        task_id.reset(tokens[4])
        sensitive_values.reset(tokens[3])
        thread_id.reset(tokens[2])
        run_id.reset(tokens[1])
        request_id.reset(tokens[0])


def execution_config(task: str, correlation: str | None, max_steps: int) -> RunnableConfig:
    """A fresh execution span; durable identity remains stable across recovery."""
    return {
        "run_name": "etl-agent",
        "run_id": uuid4(),
        "configurable": {"thread_id": task},
        "metadata": {"task_id": task, "thread_id": task, "request_id": correlation},
        "recursion_limit": max(25, max_steps * 3 + 5),
    }


def create_tracing_client(settings: Settings, *, session: Session | None = None) -> Client:
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key.get_secret_value()
        if settings.langsmith_api_key
        else None,
        timeout_ms=2000,
        session=session,
        tracing_error_callback=lambda error: logger.warning(
            "event=tracing.delivery_failed error_type=%s", type(error).__name__
        ),
    )


@asynccontextmanager
async def execution_tracing(
    settings: Settings,
    *,
    client_factory: Callable[[Settings], Client] = create_tracing_client,
) -> AsyncIterator[None]:
    client = None
    if settings.langsmith_tracing:
        try:
            client = client_factory(settings)
        except Exception as error:
            logger.warning("event=tracing.unavailable error_type=%s", type(error).__name__)
    try:
        with tracing_context(
            enabled=client is not None,
            client=client,
            project_name=settings.langsmith_project,
        ):
            yield
    finally:
        if client is not None:
            try:
                await asyncio.to_thread(client.close, timeout=2)
            except Exception as error:
                logger.warning("event=tracing.flush_failed error_type=%s", type(error).__name__)


class StateUpdate(Protocol):
    @property
    def history(self) -> list[HistoryEntry]: ...


class ObservedNode[State: RunInput, Update](Protocol):
    def __call__(self, state: State, *, runtime: Runtime[None]) -> Awaitable[Update]: ...


def observed_node[State: RunInput, Update: StateUpdate](
    name: str, function: Callable[[State, Runtime[None]], Awaitable[Update]]
) -> ObservedNode[State, Update]:
    """Instrument each attempt; summaries omit free text and full state."""

    async def invoke(state: State, runtime: Runtime[None]) -> Update:
        with node_context(get_config(), runtime, state.goal):
            attempt = runtime.execution_info.node_attempt if runtime.execution_info else 1
            started = perf_counter()
            logger.info("event=node.started node=%s attempt=%s", name, attempt)
            try:
                update = await function(state, runtime)
            except asyncio.CancelledError:
                logger.info(
                    "event=node.cancelled node=%s attempt=%s elapsed_ms=%.1f",
                    name,
                    attempt,
                    (perf_counter() - started) * 1000,
                )
                raise
            except GraphBubbleUp:
                logger.info("event=node.interrupted node=%s attempt=%s", name, attempt)
                raise
            except Exception:
                logger.exception(
                    "event=node.failed node=%s attempt=%s elapsed_ms=%.1f",
                    name,
                    attempt,
                    (perf_counter() - started) * 1000,
                )
                raise
            history = update.history
            delta = history[-1].state_delta if history else None
            # Only structural data: detailed free text remains in history/checkpoints.
            summary = {
                key: value
                for key, value in (delta or {}).items()
                if key
                in {
                    "current_step",
                    "messages_count",
                    "new_result_ids",
                    "reused_result_ids",
                    "pending_observations",
                    "last_batch_new_results",
                    "new_successful_results",
                    "consecutive_no_progress",
                    "stop_reason",
                }
            }
            for key in ("observations", "notes", "done_criteria"):
                items = (delta or {}).get(key)
                if isinstance(items, list):
                    summary[key + "_count"] = len(items)
                    if key == "done_criteria":
                        summary["satisfied_criteria_count"] = sum(
                            isinstance(item, dict) and item.get("satisfied") is True
                            for item in items
                        )
            logger.info("event=state.updated node=%s delta=%s", name, tool_log_payload(summary))
            logger.info(
                "event=node.completed node=%s attempt=%s elapsed_ms=%.1f",
                name,
                attempt,
                (perf_counter() - started) * 1000,
            )
            next_node = {
                "input": "planning",
                "planning": "acting",
                "tools": "observation",
                "finalization": "END",
            }.get(name)
            if next_node:
                logger.info("event=node.chosen node=%s next_node=%s", name, next_node)
            return update

    return invoke
