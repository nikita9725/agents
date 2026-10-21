import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SETTINGS_PROBE = """
import json
import sys
from etl_agent.settings import Settings
from pydantic import ValidationError
from pydantic_settings import SettingsError
try:
    first, second = Settings(), Settings()
    independent = first.llm_extra_body is not second.llm_extra_body
    print(json.dumps({"extra_body": first.llm_extra_body, "independent": independent}))
except (ValidationError, SettingsError):
    print("Invalid settings", file=sys.stderr)
    raise SystemExit(2)
"""


def settings_process(
    tmp_path: Path, extra_body: str | None = None
) -> subprocess.CompletedProcess[str]:
    # Only the child sees this environment; no .env or provider secrets are inherited.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "TMPDIR", "SYSTEMROOT"}
    }
    environment.update(
        LLM_MODEL="test-model",
        LLM_API_KEY="test-key",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )
    if extra_body is not None:
        environment["LLM_EXTRA_BODY"] = extra_body
    return subprocess.run(
        [sys.executable, "-c", SETTINGS_PROBE],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_extra_body_defaults_to_independent_empty_objects(tmp_path: Path) -> None:
    result = settings_process(tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"extra_body": {}, "independent": True}


def test_extra_body_reads_provider_json(tmp_path: Path) -> None:
    result = settings_process(tmp_path, '{"thinking":{"type":"disabled"}}')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("value", ["not json", "[]", '"disabled"', "true", "42", "null"])
def test_invalid_extra_body_rejects_configuration(tmp_path: Path, value: str) -> None:
    result = settings_process(tmp_path, value)
    assert result.returncode == 2
    assert "Invalid settings" in result.stderr


def test_extra_body_preserves_nested_json_values(tmp_path: Path) -> None:
    value = '{"options":[null,true,42,1.5,"value",{"thinking":{"type":"disabled"}}]}'
    result = settings_process(tmp_path, value)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["extra_body"] == json.loads(value)
