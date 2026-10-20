import os
from collections.abc import Iterator

import httpx
import pytest
from dotenv import dotenv_values

from etl_agent.startup import validate_startup
from tests.stack import ROOT, Stack


@pytest.fixture(autouse=True)
def disable_external_tracing(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.get_closest_marker("unit") is not None:
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture(scope="session")
def agent_stack() -> Iterator[Stack]:
    try:
        validate_startup()
    except ValueError as exc:
        pytest.fail(str(exc))
    stack = Stack()
    try:
        stack.execute("up")
        yield stack
    finally:
        # Stop without deleting containers/volumes, retaining evidence on failure.
        stack.execute("stop")


@pytest.fixture
def agent_client(agent_stack: Stack) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=agent_stack.url, timeout=50, trust_env=False) as client:
        yield client


def pytest_exception_interact(node: pytest.Item, call: pytest.CallInfo[object]) -> None:
    if not isinstance(node, pytest.Function) or node.get_closest_marker("unit") is not None:
        return
    stack = node.funcargs.get("agent_stack")
    if isinstance(stack, Stack):
        logs = stack.execute("logs", TAIL="200", capture=True)
        for name in ("LLM_API_KEY", "LANGSMITH_API_KEY", "LANGGRAPH_CLOUD_LICENSE_KEY"):
            value = os.environ.get(name) or dotenv_values(ROOT / ".env").get(name)
            if value:
                logs = logs.replace(value, "[REDACTED]")
        print("\nTask executor diagnostics:\n" + logs)
