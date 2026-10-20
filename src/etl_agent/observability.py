"""Контекст запроса и безопасный вывод диагностических сообщений."""

import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import TracebackType

import httpx
from dotenv import dotenv_values
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import ValidationError

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)
task_id: ContextVar[str | None] = ContextVar("task_id", default=None)
redacted_secrets: set[str] = set()
sensitive_values: ContextVar[tuple[str, ...]] = ContextVar("sensitive_values", default=())


def protect_text(*values: str) -> None:
    sensitive_values.set((*sensitive_values.get(), *(value for value in values if value)))


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
