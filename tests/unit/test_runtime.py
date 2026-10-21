from dataclasses import dataclass, field

import pytest
from pydantic import SecretStr

from etl_agent.runtime import RuntimeSettings
from etl_agent.worker import task_runtime
from tests.fakes import FakeWorkflowClient

pytestmark = pytest.mark.unit


@dataclass(slots=True)
class FakeRuntimeBackend:
    failure: str | None = None
    events: list[str] = field(default_factory=list)
    client: FakeWorkflowClient = field(default_factory=FakeWorkflowClient)
    shutdown_timeout: int | None = None
    runtime_settings: RuntimeSettings = field(
        default_factory=lambda: RuntimeSettings(
            _env_file=None, task_database_url=SecretStr("postgresql://unused/unit")
        )
    )

    def record(self, event: str) -> None:
        self.events.append(event)
        if self.failure == event:
            raise RuntimeError(event)

    async def prepare(self) -> None:
        self.record("prepare")

    def launch(self) -> None:
        self.record("launch")

    def register_queue(self) -> None:
        self.record("queue")

    def open_client(self) -> FakeWorkflowClient:
        self.record("client")
        return self.client

    def shutdown(self, *, workflow_completion_timeout_sec: int) -> None:
        self.shutdown_timeout = workflow_completion_timeout_sec
        self.record("destroy")


@pytest.mark.parametrize(
    "failure", [None, "prepare", "launch", "queue", "client", "body", "destroy"]
)
async def test_runtime_owns_resources_and_cleans_up_partial_startup(failure: str | None) -> None:
    backend = FakeRuntimeBackend(failure=failure)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            async with task_runtime(backend) as service:
                await service.ready()
                backend.record("body")
    else:
        async with task_runtime(backend) as service:
            await service.ready()
            backend.record("body")
    expected = ["prepare", "launch", "queue", "client", "body"]
    if failure and failure != "destroy":
        expected = expected[: expected.index(failure) + 1]
    if failure != "prepare":
        expected.append("destroy")
    assert backend.events == expected
    assert backend.shutdown_timeout == (None if failure == "prepare" else 25)
    assert backend.client.destroy_calls == (
        0 if failure in {"prepare", "launch", "queue", "client"} else 1
    )
