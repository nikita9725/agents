import json
from uuid import uuid4

import httpx
import pytest

from etl_agent.client import sse_events, stream_task
from etl_agent.domain import AgentState, HistoryEntry
from etl_agent.tasks import Task, TaskProgress

pytestmark = pytest.mark.unit


def test_sse_parser_handles_comments_and_multiline_data() -> None:
    assert list(
        sse_events(
            iter(
                [
                    ": keepalive",
                    "event: progress",
                    'data: {"content":',
                    'data: "plan"}',
                    "",
                ]
            )
        )
    ) == [("progress", '{"content":\n"plan"}')]


def test_client_prints_repeated_actions_and_deduplicates_replay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_id = uuid4()
    history = [
        HistoryEntry(role="assistant", node="planning", content="Read then calculate"),
        HistoryEntry(role="assistant", node="acting", content="Read log"),
        HistoryEntry(role="tool", node="observation", content="120 total, 30 rejected"),
        HistoryEntry(role="assistant", node="acting", content="Calculate 30/120*100"),
    ]
    progress = [
        TaskProgress(task_id=task_id, sequence=index, **entry.model_dump())
        for index, entry in enumerate(history, 1)
    ]
    finished = Task(
        task_id=task_id,
        status="success",
        result=AgentState(
            goal="Calculate",
            history=history,
            final_answer="25%",
        ),
    )
    body = (
        "".join(
            f"event: progress\ndata: {entry.model_dump_json()}\n\n"
            for entry in [*progress[:2], *progress]
        )
        + f"event: task\ndata: {finished.model_dump_json()}\n\n"
    )
    with httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
    ) as client:
        assert stream_task(client, task_id) == finished
    output = capsys.readouterr().out
    assert output.count("action:") == 2
    assert output.index("plan:") < output.index("Read log") < output.index("observation:")
    assert output.count("Read log") == 1


@pytest.mark.parametrize("case", ["error", "disconnect", "timeout"])
def test_client_surfaces_stream_failure(case: str) -> None:
    task_id = uuid4()
    body = ""
    if case == "error":
        body = (
            "event: error\ndata: "
            + json.dumps(
                {
                    "task_id": str(task_id),
                    "error": {"code": "executor_unavailable", "message": "Reconnect"},
                }
            )
            + "\n\n"
        )
    elif case == "timeout":
        body = f"event: task\ndata: {Task(task_id=task_id, status='running').model_dump_json()}\n\n"
    with (
        httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
        ) as client,
        pytest.raises(TimeoutError if case == "timeout" else RuntimeError),
    ):
        stream_task(client, task_id, timeout=0 if case == "timeout" else 300)
