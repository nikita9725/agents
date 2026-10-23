"""Создать задачу через API и показать plan → action → observation через SSE."""

import argparse
import json
import time
from collections.abc import Iterator
from uuid import UUID

import httpx

from etl_agent.tasks import TERMINAL, Task, TaskProgress, TaskStreamError


def stream_task(client: httpx.Client, task_id: UUID, *, timeout: float = 300) -> Task:
    deadline = time.monotonic() + timeout
    seen: set[int] = set()
    with client.stream("GET", f"/tasks/{task_id}/events", timeout=timeout) as response:
        response.raise_for_status()
        for event, data in sse_events(response.iter_lines()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Task did not finish within the observation timeout.")
            if event == "progress":
                progress = TaskProgress.model_validate_json(data)
                if progress.sequence in seen:
                    continue
                seen.add(progress.sequence)
                label = {"planning": "plan", "acting": "action"}.get(progress.node, progress.node)
                print(f"[{progress.sequence}] {label}: {progress.content}", flush=True)
            elif event == "error":
                error = TaskStreamError.model_validate_json(data)
                raise RuntimeError(error.error.message)
            elif event == "task":
                task = Task.model_validate_json(data)
                if task.status in TERMINAL:
                    return task
    raise RuntimeError("Event stream closed before task completion; reconnect to observe the task.")


def sse_events(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    event = "message"
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal")
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    args = parser.parse_args()
    try:
        with httpx.Client(base_url=args.url, timeout=50, trust_env=False) as client:
            response = client.post("/tasks", json={"goal": args.goal})
            response.raise_for_status()
            task = Task.model_validate(response.json())
            print(f"task_id={task.task_id}", flush=True)
            task = stream_task(client, task.task_id)
            if task.status != "success":
                parser.exit(1, f"Task ended: {task.model_dump_json()}\n")
            assert task.result is not None
            print(json.dumps(task.result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    except (httpx.HTTPError, TimeoutError, RuntimeError) as exc:
        parser.exit(1, f"Could not observe task: {exc}\n")
