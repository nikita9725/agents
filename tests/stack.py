"""Запуск тестового стека и управление контейнерами для реальных e2e."""

import json
import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ContainerDeath:
    exit_code: str
    time_ns: int


class Stack:
    def __init__(self) -> None:
        self._environment = {
            **{
                key: value
                for key, value in dotenv_values(ROOT / ".env").items()
                if value is not None
            },
            **os.environ,
        }
        self._port = int(self._environment.get("AGENT_TEST_PORT", "18123"))
        self.url = f"http://127.0.0.1:{self._port}"

    def execute(
        self,
        target: str,
        *,
        capture: bool = False,
        check: bool = True,
        **variables: str,
    ) -> str:
        compose = [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "-p",
            "etl-agent-test",
            "-f",
            str(ROOT / "compose.yaml"),
        ]
        match target:
            case "up":
                command = ["make", "--no-print-directory", "up", "COMPOSE_PROJECT=etl-agent-test"]
            case "stop":
                command = [*compose, "stop"]
            case "recreate-api":
                command = [*compose, "up", "-d", "--no-deps", "--force-recreate", "api"]
            case "runtime-check":
                command = [
                    *compose,
                    "exec",
                    "-T",
                    "api",
                    "python",
                    "-m",
                    "tests.runtime_check",
                    variables["TASK_ID"],
                    variables["ACTION"],
                ]
            case "logs":
                command = [*compose, "logs", "--no-color"]
                if variables.get("UNTIL"):
                    command.extend(["--until", variables["UNTIL"]])
                if variables.get("TAIL"):
                    command.extend(["--tail", variables["TAIL"]])
                command.append("api")
            case "container":
                command = [*compose, "ps", "-q", "api"]
            case "restart-count":
                command = [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.RestartCount}}",
                    variables["CONTAINER"],
                ]
            case "crash":
                command = [*compose, "exec", "-T", "api", "python", "-"]
            case _:
                raise ValueError(f"Unknown test stack operation: {target}")
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=check,
            text=True,
            input=(ROOT / "tests/crash.py").read_text() if target == "crash" else None,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            env={
                **self._environment,
                "AGENT_PORT": str(self._port),
                "POSTGRES_PORT": self._environment.get("POSTGRES_TEST_PORT", "15432"),
                **variables,
            },
        )
        return result.stdout or ""

    def crash_and_observe(self, container: str) -> ContainerDeath:
        # Subscribe before the signal; do not rely on a short retrospective time window.
        command = [
            "docker",
            "events",
            "--since",
            str(time.time() - 1),
            "--filter",
            "container=" + container,
            "--filter",
            "event=die",
            "--format",
            "{{json .}}",
        ]
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=self._environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as observer:
            try:
                assert observer.stdout is not None
                with selectors.DefaultSelector() as selector:
                    selector.register(observer.stdout, selectors.EVENT_READ)
                    crash_output = self.execute("crash", check=False, capture=True)
                    assert selector.select(timeout=30), (
                        "No Docker die event after SIGKILL",
                        crash_output,
                    )
                    line = observer.stdout.readline()
                    assert line, ("Docker event observer exited", crash_output)
                    event = json.loads(line)
                    return ContainerDeath(
                        exit_code=event["Actor"]["Attributes"]["exitCode"],
                        time_ns=int(event["timeNano"]),
                    )
            finally:
                observer.terminate()
                try:
                    observer.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    observer.kill()
                    observer.communicate()
