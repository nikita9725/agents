import httpx
import pytest

from etl_agent.tasks import Task
from tests.runtime_check import RuntimeInfo
from tests.stack import Stack
from tests.tasks import submit, successful_state

pytestmark = pytest.mark.integration


def test_real_provider_persistence_and_independent_tasks(
    agent_client: httpx.Client, agent_stack: Stack
) -> None:
    first_id = submit(agent_client, "Учебная задача: входной файл sales.csv отсутствует")
    second_id = submit(agent_client, "Учебная задача: permission denied при чтении")
    assert first_id != second_id
    first = successful_state(agent_client, first_id)
    second = successful_state(agent_client, second_id)
    assert first.goal != second.goal
    saved = Task.model_validate(agent_client.get(f"/tasks/{first_id}").json())
    assert saved.result == first
    info = RuntimeInfo.model_validate_json(
        agent_stack.execute(
            "runtime-check",
            TASK_ID=str(first_id),
            ACTION="inspect",
            capture=True,
        )
    )
    assert info.history_count >= 3 and info.run_count == 1
