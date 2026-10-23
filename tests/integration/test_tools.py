import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.tasks import submit, successful_state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "name,goal,expected",
    [
        (
            "calculator",
            "Используй calculator для расчёта (120-90)/120*100. Укажи процент отклонённых строк.",
            "25",
        ),
        (
            "read_local_file",
            "Прочитай etl.log инструментом read_local_file. Сколько строк отклонено?",
            "30",
        ),
        (
            "text_search",
            "Используй text_search для поиска строки invalid date в etl.log. Какое значение даты вызвало ошибку?",
            "2026-99-21",
        ),
    ],
)
def test_real_tool_choice_and_observation(
    agent_client: httpx.Client, name: str, goal: str, expected: str
) -> None:
    state = successful_state(agent_client, submit(agent_client, goal))
    calls = {
        call["id"]: call["name"]
        for message in state.messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    observations = [
        message
        for message in state.messages
        if isinstance(message, ToolMessage) and calls.get(message.tool_call_id) == name
    ]
    assert observations and all(message.status == "success" for message in observations)
    assert expected in json.dumps([message.content for message in observations], ensure_ascii=False)
    assert state.final_answer and expected in state.final_answer


def test_real_read_then_calculate(agent_client: httpx.Client) -> None:
    state = successful_state(
        agent_client,
        submit(
            agent_client,
            "Сначала read_local_file прочитай etl.log. Только после чтения calculator вычисли процент отклонённых строк из expected_rows и rejected_rows. Укажи этот процент.",
        ),
    )
    messages = state.messages
    read_index = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
        and message.name == "read_local_file"
        and message.status == "success"
    )
    calculation_index = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        and any(call["name"] == "calculator" for call in message.tool_calls)
    )
    assert read_index < calculation_index
    assert state.plan and state.plan.steps
    nodes = [entry.node for entry in state.history]
    assert nodes.count("planning") == 1
    assert nodes.count("acting") >= 2 and nodes.count("observation") >= 2
    assert nodes.index("observation") < max(i for i, node in enumerate(nodes) if node == "acting")
    assert state.final_answer and "25" in state.final_answer
