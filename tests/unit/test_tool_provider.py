import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from etl_agent.domain import RunInput
from etl_agent.infrastructure.graph import build_graph
from etl_agent.infrastructure.planner import OpenAIPlanner
from etl_agent.infrastructure.tools import create_tools
from tests.unit.test_planner import completion, completion_json

pytestmark = pytest.mark.unit


async def test_provider_receives_schemas_and_observation(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload["tools"][0]["function"]["name"] == "ExecutionPlan"
            assert "calculator" in payload["messages"][0]["content"]
            assert '"steps":["' in payload["messages"][0]["content"]
            assert (
                payload["tools"][0]["function"]["parameters"]["properties"]["steps"]["items"][
                    "type"
                ]
                == "string"
            )
            body = completion('{"steps":["Calculate rejected rows"]}')
            assert body.choices[0].message.tool_calls
            body.choices[0].message.tool_calls[0].function.name = "ExecutionPlan"
        elif len(requests) == 2:
            assert "Calculate rejected rows" in payload["messages"][-1]["content"]
            assert {tool["function"]["name"] for tool in payload["tools"]} == {
                "calculator",
                "read_local_file",
                "text_search",
            }
            body = completion('{"expression":"120-90"}')
            assert body.choices[0].message.tool_calls
            body.choices[0].message.tool_calls[0].function.name = "calculator"
        elif len(requests) == 3:
            assert payload["messages"][-1]["role"] == "tool"
            assert payload["messages"][-1]["tool_call_id"] == "call_test"
            assert json.loads(payload["messages"][-1]["content"]) == {"result": 30.0}
            body = completion("{}")
            body.choices[0].message.tool_calls = None
            body.choices[0].message.content = "30 rows rejected"
            body.choices[0].finish_reason = "stop"
        else:
            assert payload["tools"][0]["function"]["name"] == "FinalResponse"
            assert any(message["role"] == "tool" for message in payload["messages"])
            body = completion(
                json.dumps(
                    {
                        "final_answer": "30 rows rejected",
                        "next_action": {
                            "action": "Inspect rejected rows",
                            "rationale": "30 rows were rejected",
                        },
                    }
                )
            )
            assert body.choices[0].message.tool_calls
            body.choices[0].message.tool_calls[0].function.name = "FinalResponse"
        from dataclasses import asdict

        return httpx.Response(200, json=asdict(body))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        chat = ChatOpenAI(model="test", api_key=SecretStr("key"), http_async_client=client)
        tools = create_tools(tmp_path)
        planner = OpenAIPlanner(
            base_url="https://provider.invalid/v1",
            model="test",
            api_key=SecretStr("key"),
            client=chat,
            tools=tools,
        )
        result = await build_graph(planner, tools=tools).ainvoke(
            RunInput(goal="Calculate rejected rows")
        )
    assert len(requests) == 4
    assert result["final_answer"] == "30 rows rejected"


async def test_malformed_tool_call_is_rejected(tmp_path: Path) -> None:
    from etl_agent.exceptions import PlanningError

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_json("not-json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        planner = OpenAIPlanner(
            base_url="https://provider.invalid/v1",
            model="test",
            api_key=SecretStr("key"),
            client=ChatOpenAI(model="test", api_key=SecretStr("key"), http_async_client=client),
            tools=create_tools(tmp_path),
        )
        with pytest.raises(PlanningError):
            await planner.act([HumanMessage(content="Inspect")])


async def test_finalization_http_error_is_classified(tmp_path: Path) -> None:
    from etl_agent.exceptions import TransientPlanningError

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": {"message": "busy"}})
        )
    ) as client:
        planner = OpenAIPlanner(
            base_url="https://provider.invalid/v1",
            model="test",
            api_key=SecretStr("key"),
            client=ChatOpenAI(
                model="test", api_key=SecretStr("key"), http_async_client=client, max_retries=0
            ),
            tools=create_tools(tmp_path),
        )
        with pytest.raises(TransientPlanningError):
            await planner.finalize(
                [
                    HumanMessage(content="Inspect"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "calculator", "args": {"expression": "1+1"}, "id": "one"}
                        ],
                    ),
                    ToolMessage(content='{"result":2}', tool_call_id="one"),
                ]
            )


@pytest.mark.parametrize("arguments", ['{"steps":[]}', '{"steps":[" "]}', "not json"])
async def test_invalid_execution_plan_is_rejected(arguments: str) -> None:
    from dataclasses import asdict

    from etl_agent.exceptions import PlanningError

    def handle(request: httpx.Request) -> httpx.Response:
        body = completion(arguments)
        assert body.choices[0].message.tool_calls
        body.choices[0].message.tool_calls[0].function.name = "ExecutionPlan"
        return httpx.Response(200, json=asdict(body))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        planner = OpenAIPlanner(
            base_url="https://provider.invalid/v1",
            model="test",
            api_key=SecretStr("key"),
            client=ChatOpenAI(model="test", api_key=SecretStr("key"), http_async_client=client),
        )
        with pytest.raises(PlanningError):
            await planner.create_plan("Inspect")
