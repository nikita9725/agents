import json
from dataclasses import asdict, dataclass
from unittest.mock import patch

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import JsonValue, SecretStr

from etl_agent.exceptions import PlanningError, PlanningTimeout
from etl_agent.infrastructure.planner import OpenAIPlanner

pytestmark = pytest.mark.unit


@dataclass(slots=True)
class Function:
    name: str
    arguments: str


@dataclass(slots=True)
class ToolCall:
    id: str
    type: str
    function: Function


@dataclass(slots=True)
class Message:
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None


@dataclass(slots=True)
class Choice:
    index: int
    finish_reason: str
    message: Message


@dataclass(slots=True)
class Completion:
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]


def completion(arguments: str) -> Completion:
    return Completion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="test-model",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_test",
                            type="function",
                            function=Function(name="NextAction", arguments=arguments),
                        )
                    ],
                ),
            )
        ],
    )


def completion_json(arguments: str) -> dict[str, object]:
    return asdict(completion(arguments))


def planner_with_transport(client: httpx.AsyncClient) -> OpenAIPlanner:
    chat = ChatOpenAI(
        model="test-model",
        base_url="https://provider.invalid/v1",
        api_key=SecretStr("test-key"),
        http_async_client=client,
        timeout=30,
        max_retries=0,
    )
    return OpenAIPlanner(
        base_url="https://provider.invalid/v1",
        model="test-model",
        api_key=SecretStr("test-key"),
        client=chat,
    )


async def test_structured_function_call_and_user_input() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=completion_json(
                json.dumps(
                    {
                        "action": "Read the failed run log",
                        "rationale": "Need the precise error",
                    }
                )
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await planner_with_transport(client).propose("Sales load failed")
    assert result.action == "Read the failed run log"
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert payload["tools"][0]["function"]["name"] == "NextAction"
    assert payload["messages"][-1]["content"] == "Sales load failed"


@pytest.mark.parametrize(
    "arguments", ['{"action":"Read logs"}', "not json", '{"action":"","rationale":"x"}']
)
async def test_invalid_provider_output(arguments: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=completion_json(arguments))
        )
    ) as client:
        with pytest.raises(PlanningError):
            await planner_with_transport(client).propose("ETL failed")


@pytest.mark.parametrize("failure", ["timeout", "server_error", "no_tool_call"])
async def test_provider_failures_are_classified_without_retries(failure: str) -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("private details", request=request)
        if failure == "server_error":
            return httpx.Response(500, json={"error": {"message": "private details"}})
        body = completion("{}")
        body.choices[0].message = Message(role="assistant", content="unstructured")
        body.choices[0].finish_reason = "stop"
        return httpx.Response(200, json=asdict(body))

    error: type[PlanningTimeout] | type[PlanningError] = (
        PlanningTimeout if failure == "timeout" else PlanningError
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(error):
            await planner_with_transport(client).propose("ETL failed")
    assert calls == 1


@pytest.mark.parametrize("extra_body", [None, {"thinking": {"type": "disabled"}}])
async def test_extra_body_reaches_provider_without_changing_function_contract(
    extra_body: dict[str, JsonValue] | None,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=completion_json(
                json.dumps(
                    {
                        "action": "Read logs",
                        "rationale": "Need the error",
                    }
                )
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport_client:
        # Exercise the planner's production construction path with an offline SDK transport.
        with patch("etl_agent.infrastructure.planner.ChatOpenAI", wraps=ChatOpenAI) as factory:
            factory.side_effect = lambda **kwargs: ChatOpenAI(
                **kwargs,
                http_async_client=transport_client,
            )
            planner = OpenAIPlanner(
                base_url="https://provider.invalid/v1",
                model="test-model",
                api_key=SecretStr("test-key"),
                extra_body=extra_body,
            )
            result = await planner.propose("Missing input file")
    assert result.action == "Read logs"
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    if extra_body:
        assert payload["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in payload
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "NextAction"}}
    assert payload["tools"][0]["function"]["strict"] is True
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["action", "rationale"]


async def test_injected_client_keeps_its_configuration() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=completion_json(
                json.dumps(
                    {
                        "action": "Read logs",
                        "rationale": "Need the error",
                    }
                )
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport_client:
        chat = ChatOpenAI(
            model="test-model",
            base_url="https://provider.invalid/v1",
            api_key=SecretStr("test-key"),
            http_async_client=transport_client,
            extra_body={"thinking": {"type": "disabled"}},
        )
        planner = OpenAIPlanner(
            base_url="https://provider.invalid/v1",
            model="test-model",
            api_key=SecretStr("test-key"),
            client=chat,
            extra_body={"other_parameter": True},
        )
        await planner.propose("Missing input file")
    payload = json.loads(requests[0].content)
    assert payload["thinking"] == {"type": "disabled"}
    assert "other_parameter" not in payload


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_only_transient_http_errors_are_retryable(status: int) -> None:
    from etl_agent.exceptions import TransientPlanningError

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "provider failure"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(PlanningError) as error:
            await planner_with_transport(client).propose("missing file")
    assert isinstance(error.value, TransientPlanningError) is (status == 429 or status >= 500)
    assert error.value.__cause__ is not None
