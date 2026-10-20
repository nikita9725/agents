"""Клиентские проверки единственного публичного API задач."""

import time
from uuid import UUID

import httpx

from etl_agent.domain import AgentState
from etl_agent.tasks import Task


def submit(client: httpx.Client, goal: str) -> UUID:
    response = client.post("/tasks", json={"goal": goal})
    assert response.status_code == 202, response.text
    assert "thread_id" not in response.text and "run_id" not in response.text
    task = Task.model_validate(response.json())
    assert response.headers["Location"] == f"/tasks/{task.task_id}"
    return task.task_id


def wait_task(client: httpx.Client, task_id: UUID, *, timeout: float = 150) -> Task:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/tasks/{task_id}")
        assert response.is_success, response.text
        task = Task.model_validate(response.json())
        if task.status in {"success", "error", "timeout", "interrupted"}:
            return task
        time.sleep(0.1)
    raise AssertionError(f"Task did not finish: task_id={task_id}")


def successful_state(client: httpx.Client, task_id: UUID) -> AgentState:
    task = wait_task(client, task_id)
    assert task.status == "success", task
    assert task.result is not None
    state = task.result
    assert state.next_action is not None and state.final_answer
    assert state.current_step == 1
    assert [entry.node for entry in state.history] == ["input", "planning", "finalization"]
    return state
