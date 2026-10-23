import json
import re
import time
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from etl_agent.infrastructure.tools import CalculatorOutput, FileOutput, SearchOutput
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
    task_id = submit(agent_client, "Используй calculator для вычисления (120-90)/120*100")
    with agent_client.stream("GET", f"/tasks/{task_id}/events", timeout=190) as response:
        assert response.status_code == 200
        events = list(response.iter_lines())
    text = "\n".join(events)
    assert "thread_id" not in text and "run_id" not in text
    data = [json.loads(line[6:]) for line in events if line.startswith("data: ")]
    assert {entry.get("node") for entry in data if "node" in entry} == {
        "input",
        "planning",
        "acting",
        "observation",
        "finalization",
    }
    progress = [entry for entry in data if "node" in entry]
    assert [entry["sequence"] for entry in progress] == list(range(1, len(progress) + 1))
    assert all(entry["content"] and entry["role"] for entry in progress)
    assert sum(entry["node"] == "acting" for entry in progress) >= 2
    result = successful_state(agent_client, task_id)
    assert [entry["node"] for entry in progress] == [entry.node for entry in result.history]
    assert any(entry.get("status") == "success" for entry in data), data
    replay = agent_client.get(f"/tasks/{task_id}/events")
    replay_data = [
        json.loads(line[6:]) for line in replay.text.splitlines() if line.startswith("data: ")
    ]
    assert [entry for entry in replay_data if "node" in entry] == progress


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

    death = agent_stack.crash_and_observe(container)
    assert death.exit_code == "137", death
    death_ns = death.time_ns
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
) -> None:
    try:
        agent_stack.execute("recreate-api", LLM_API_KEY="invalid-native-e2e-verification")
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


def test_full_graph_reads_searches_and_calculates(agent_client: httpx.Client) -> None:
    state = successful_state(
        agent_client,
        submit(
            agent_client,
            (
                "Прочитай etl.log инструментом read_local_file и найди invalid date через text_search. "
                "После получения содержимого файла вызови calculator для расчёта процента "
                "отклонённых строк из expected_rows и rejected_rows. "
                "В итоговом ответе укажи полученный процент числом со знаком %."
            ),
        ),
    )
    nodes = [entry.node for entry in state.history]
    assert {"input", "planning", "acting", "observation", "finalization"} <= set(nodes)
    assert (
        nodes.index("observation")
        < max(i for i, node in enumerate(nodes) if node == "acting")
        < len(nodes) - 1
    )

    calls = {
        call["id"]: (index, call["name"])
        for index, message in enumerate(state.messages)
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    observations: dict[str, list[tuple[int, int, ToolMessage]]] = {}
    for index, message in enumerate(state.messages):
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue
        assert message.tool_call_id in calls, message
        call_index, name = calls[message.tool_call_id]
        assert call_index < index
        observations.setdefault(name, []).append((call_index, index, message))
    assert {"read_local_file", "text_search", "calculator"} <= observations.keys()

    read_indices: list[int] = []
    for _, index, message in observations["read_local_file"]:
        assert isinstance(message.content, str)
        output = FileOutput.model_validate_json(message.content)
        if "expected_rows=120" in output.content and "rejected_rows=30" in output.content:
            read_indices.append(index)
    assert read_indices, "No successful observation contains the ETL row counts"

    matches = []
    for _, _, message in observations["text_search"]:
        assert isinstance(message.content, str)
        matches.extend(SearchOutput.model_validate_json(message.content).matches)
    assert any("invalid date" in match.text for match in matches)

    calculation_indices: list[int] = []
    for call_index, _, message in observations["calculator"]:
        assert isinstance(message.content, str)
        result = CalculatorOutput.model_validate_json(message.content).result
        if result == pytest.approx(25.0):
            calculation_indices.append(call_index)
    assert any(
        read < calculation for read in read_indices for calculation in calculation_indices
    ), "Expected a calculation of 25% after reading row counts"
    assert state.final_answer is not None
    percentages = [
        float(number.replace(",", "."))
        for number in re.findall(r"(?<![\d.,+−-])(\d+(?:[.,]\d+)?)\s*%", state.final_answer)
    ]
    assert any(value == pytest.approx(25.0) for value in percentages), state.final_answer
    assert state.next_action is not None
    assert state.next_action.action.strip() and state.next_action.rationale.strip()
