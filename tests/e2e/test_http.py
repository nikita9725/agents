import json
import time
from uuid import UUID, uuid4

import httpx
import pytest

from etl_agent.tasks import Task
from tests.runtime_check import RuntimeInfo
from tests.stack import Stack
from tests.tasks import submit, successful_state, wait_task

pytestmark = pytest.mark.e2e


def ready(stack: Stack, *, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(stack.url + "/ready").is_success:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.1)
    raise AssertionError("Application did not recover: " + stack.url)


def test_result_survives_container_recreation(
    agent_stack: Stack, agent_client: httpx.Client
) -> None:
    task_id = submit(agent_client, "Учебная задача: не найден файл sales.csv")
    original = successful_state(agent_client, task_id)
    agent_stack.execute("recreate-api")
    ready(agent_stack)
    saved = Task.model_validate(agent_client.get(f"/tasks/{task_id}").json())
    assert saved.status == "success" and saved.result == original


def test_public_streaming(agent_client: httpx.Client) -> None:
    task_id = submit(agent_client, "Учебная задача: неверный формат даты CSV")
    with agent_client.stream("GET", f"/tasks/{task_id}/events", timeout=190) as response:
        assert response.status_code == 200
        events = list(response.iter_lines())
    text = "\n".join(events)
    assert "thread_id" not in text and "run_id" not in text
    data = [json.loads(line[6:]) for line in events if line.startswith("data: ")]
    assert {entry.get("node") for entry in data if "node" in entry} == {
        "input",
        "planning",
        "finalization",
    }
    assert any(entry.get("status") == "success" for entry in data), data


def runtime_info(stack: Stack, task_id: UUID, *, action: str = "inspect") -> RuntimeInfo:
    return RuntimeInfo.model_validate_json(
        stack.execute(
            "runtime-check",
            TASK_ID=str(task_id),
            ACTION=action,
            capture=True,
        )
    )


def test_queued_task_and_cancellation_survive_restart(
    agent_stack: Stack, agent_client: httpx.Client
) -> None:
    # Prepare deferred native work inside Docker; the public API has no scheduling/debug fields.
    queued, cancelled = uuid4(), uuid4()
    runtime_info(agent_stack, queued, action="defer")
    runtime_info(agent_stack, cancelled, action="defer")
    assert agent_client.get(f"/tasks/{queued}").json()["status"] == "pending"
    response = agent_client.post(f"/tasks/{cancelled}/cancel")
    assert response.is_success, response.text
    agent_stack.execute("recreate-api")
    ready(agent_stack)
    successful_state(agent_client, queued)
    assert wait_task(agent_client, cancelled).status == "interrupted"
    assert runtime_info(agent_stack, queued).run_count == 1


def test_sigkill_recovers_same_run_without_resubmission(
    agent_stack: Stack,
    agent_client: httpx.Client,
) -> None:
    # Docker only treats a container as successfully started after 10 seconds.
    time.sleep(10)
    container = agent_stack.execute("container", capture=True).strip()
    assert container

    def restart_count() -> int:
        return int(agent_stack.execute("restart-count", CONTAINER=container, capture=True).strip())

    before = restart_count()
    task_id = submit(agent_client, "Учебная задача: загрузка продаж упала, файла нет")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        info = runtime_info(agent_stack, task_id)
        assert info.status not in {"success", "error", "timeout"}, (
            "Task finished before SIGKILL; recovery has not been tested",
            info,
        )
        if info.status == "running" and info.has_input:
            assert not info.has_action, "Planning completed before crash; invalid scenario"
            break
        time.sleep(0.01)
    else:
        pytest.fail("No active task with a durable input checkpoint was observed")
    run_id = info.run_id

    kill_started = time.time()
    agent_stack.execute("crash", check=False, capture=True)
    events = agent_stack.execute(
        "events",
        SINCE=str(int(kill_started) - 1),
        UNTIL=str(int(time.time()) + 1),
        CONTAINER=container,
        capture=True,
    )
    deaths = [json.loads(line) for line in events.splitlines() if line.strip()]
    assert deaths and deaths[-1]["Actor"]["Attributes"]["exitCode"] == "137", events
    death_ns = int(deaths[-1]["timeNano"])
    death_time = f"{death_ns // 1_000_000_000}.{death_ns % 1_000_000_000:09d}"
    logs = agent_stack.execute("logs", UNTIL=death_time, capture=True)
    matching = [line for line in logs.splitlines() if run_id in line]
    assert any("event=node.started node=planning" in line for line in matching), logs
    assert not any("event=node.completed node=planning" in line for line in matching), (
        "Planning completed before SIGKILL; recovery scenario invalid",
        matching,
    )
    ready(agent_stack)
    assert restart_count() > before, "Server was not automatically restarted after SIGKILL"
    result = wait_task(agent_client, task_id, timeout=300)
    assert result.status == "success", result
    successful_state(agent_client, task_id)
    recovered = runtime_info(agent_stack, task_id)
    assert recovered.run_id == run_id and recovered.run_count == 1


def test_real_invalid_key_is_a_terminal_error(
    agent_stack: Stack,
    agent_client: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        with monkeypatch.context() as patch:
            patch.setenv("LLM_API_KEY", "invalid-native-e2e-verification")
            agent_stack.execute("recreate-api")
            ready(agent_stack)
            task_id = submit(agent_client, "Учебная проверка неверного ключа")
            result = wait_task(agent_client, task_id)
            assert result.status == "error", result
            assert result.error is not None and result.error.code == "execution_failed"
            logs = agent_stack.execute("logs", capture=True)
            matching = [line for line in logs.splitlines() if str(task_id) in line]
            assert sum("event=node.started node=planning" in line for line in matching) == 1
    finally:
        agent_stack.execute("recreate-api")
        ready(agent_stack)


def test_only_tasks_api_is_public(agent_stack: Stack, agent_client: httpx.Client) -> None:
    schema = agent_client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/ok",
        "/ready",
        "/tasks",
        "/tasks/{task_id}",
        "/tasks/{task_id}/cancel",
        "/tasks/{task_id}/events",
    }
    for path in ("/threads", "/runs", "/assistants", "/store"):
        assert agent_client.get(path).status_code == 404
    assert agent_client.get("/ok").is_success
