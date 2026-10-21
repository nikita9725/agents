import json
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from etl_agent.infrastructure.tools import MAX_BYTES, create_tools

pytestmark = pytest.mark.unit


def invoke(root: Path, name: str, args: dict[str, object]) -> ToolMessage:
    tool = next(tool for tool in create_tools(root) if tool.name == name)
    result = tool.invoke({"name": name, "args": args, "id": "call_test", "type": "tool_call"})
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_test"
    return result


@pytest.mark.parametrize("expression, expected", [("(120-90)/120*100", 25), ("-2 + +3", 1)])
def test_calculator(tmp_path: Path, expression: str, expected: float) -> None:
    message = invoke(tmp_path, "calculator", {"expression": expression})
    assert message.status == "success"
    assert json.loads(str(message.content)) == {"result": expected}


@pytest.mark.parametrize(
    "expression",
    ["1/0", "2**100", "__import__('os')", "True", "1e999", "(", "1+" * 100 + "1", "1" * 513],
)
def test_invalid_calculation_is_observation(tmp_path: Path, expression: str) -> None:
    message = invoke(tmp_path, "calculator", {"expression": expression})
    assert message.status == "error"
    assert "error" in json.loads(str(message.content))


def test_read_and_search(tmp_path: Path) -> None:
    (tmp_path / "etl.log").write_text("INFO старт\nERROR ошибка\nERROR дата\n")
    read = invoke(tmp_path, "read_local_file", {"path": "etl.log"})
    assert json.loads(str(read.content))["content"].startswith("INFO старт")
    search = invoke(tmp_path, "text_search", {"path": "etl.log", "query": "ERROR"})
    assert [item["line"] for item in json.loads(str(search.content))["matches"]] == [2, 3]
    empty = invoke(tmp_path, "text_search", {"path": "etl.log", "query": "error"})
    assert json.loads(str(empty.content))["matches"] == []


def test_paths_and_encoding(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "secret").write_text("private")
    (root / "link").symlink_to(tmp_path / "secret")
    (root / "binary").write_bytes(b"\xff")
    for path in ["../secret", str(tmp_path / "secret"), "link", "missing", ".", "binary"]:
        message = invoke(root, "read_local_file", {"path": path})
        assert message.status == "error"
        assert "private" not in str(message.content)


def test_bounds(tmp_path: Path) -> None:
    (tmp_path / "large").write_bytes(b"a" * (MAX_BYTES - 1) + "я".encode() + b"z")
    result = invoke(tmp_path, "read_local_file", {"path": "large"})
    body = json.loads(str(result.content))
    assert body["truncated"] is True
    assert len(body["content"]) == MAX_BYTES - 1
    (tmp_path / "many").write_text("ERROR\n" * 60)
    result = invoke(tmp_path, "text_search", {"path": "many", "query": "ERROR"})
    body = json.loads(str(result.content))
    assert len(body["matches"]) == 50 and body["truncated"]


@pytest.mark.parametrize(
    "name,args",
    [
        ("calculator", {}),
        ("read_local_file", {"path": ""}),
        ("text_search", {"path": "x", "query": ""}),
        ("calculator", {"expression": "1", "extra": 1}),
    ],
)
def test_schema_errors(tmp_path: Path, name: str, args: dict[str, object]) -> None:
    assert invoke(tmp_path, name, args).status == "error"
