"""Создать задачу через публичный FastAPI API и дождаться результата."""

import argparse
import time

import httpx

from etl_agent.tasks import Task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal")
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=50, trust_env=False) as client:
        response = client.post("/tasks", json={"goal": args.goal})
        response.raise_for_status()
        task = Task.model_validate(response.json())
        print(f"task_id={task.task_id}", flush=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            response = client.get(f"/tasks/{task.task_id}")
            response.raise_for_status()
            task = Task.model_validate(response.json())
            if task.status == "success":
                assert task.result is not None
                print(task.result.model_dump_json(indent=2))
                return
            if task.status in {"error", "timeout", "interrupted"}:
                parser.exit(1, f"Task ended: {task.model_dump_json()}\n")
            time.sleep(0.2)
    parser.exit(1, "Task did not finish within 300 seconds.\n")


if __name__ == "__main__":
    main()
