"""Запуск тестового стека и управление контейнерами для реальных e2e."""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


class Stack:
    def __init__(self) -> None:
        load_dotenv(ROOT / ".env")
        self._port = int(os.environ.get("AGENT_TEST_PORT", "18123"))
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
            case "events":
                command = [
                    "docker",
                    "events",
                    "--since",
                    variables["SINCE"],
                    "--until",
                    variables["UNTIL"],
                    "--filter",
                    "container=" + variables["CONTAINER"],
                    "--filter",
                    "event=die",
                    "--format",
                    "{{json .}}",
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
                **os.environ,
                "AGENT_PORT": str(self._port),
                "POSTGRES_PORT": os.environ.get("POSTGRES_TEST_PORT", "15432"),
                **variables,
            },
        )
        return result.stdout or ""
