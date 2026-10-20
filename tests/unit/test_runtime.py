from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from etl_agent.worker import task_runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure", [None, "launch", "queue", "client", "body"])
async def test_runtime_owns_resources_and_cleans_up_partial_startup(failure: str | None) -> None:
    events: list[str] = []

    def record(name: str) -> None:
        events.append(name)
        if failure == name:
            raise RuntimeError(name)

    @asynccontextmanager
    async def saver() -> AsyncIterator[Mock]:
        yield Mock(setup=AsyncMock())

    client = Mock(destroy=lambda: record("client.destroy"))
    with (
        patch("etl_agent.worker.configure_logging"),
        patch("etl_agent.worker.checkpoints", side_effect=lambda settings: saver()),
        patch("etl_agent.worker.DBOS") as dbos,
        patch("etl_agent.worker.DBOSClient") as factory,
    ):
        dbos.launch.side_effect = lambda: record("launch")
        dbos.register_queue.side_effect = lambda *args, **kwargs: record("queue")
        dbos.destroy.side_effect = lambda **kwargs: record("destroy")

        def create_client(**kwargs: object) -> Mock:
            record("client")
            return client

        factory.side_effect = create_client
        if failure:
            with pytest.raises(RuntimeError, match=failure):
                async with task_runtime():
                    record("body")
        else:
            async with task_runtime():
                record("body")
        expected = ["launch", "queue", "client", "body"]
        if failure:
            expected = expected[: expected.index(failure) + 1]
        expected.append("destroy")
        if failure not in {"launch", "queue", "client"}:
            expected.append("client.destroy")
        assert events == expected
        dbos.destroy.assert_called_once_with(workflow_completion_timeout_sec=25)
