import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsError

from etl_agent.settings import Settings

pytestmark = pytest.mark.unit


def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_model="test-model",
        llm_api_key=SecretStr("test-key"),
    )


def test_extra_body_defaults_to_independent_empty_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    first, second = settings(), settings()
    assert first.llm_extra_body == second.llm_extra_body == {}
    first.llm_extra_body["thinking"] = {"type": "disabled"}
    assert second.llm_extra_body == {}


def test_extra_body_reads_provider_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    assert settings().llm_extra_body == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("value", ["not json", "[]", '"disabled"', "true", "42", "null"])
def test_invalid_extra_body_rejects_configuration(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("LLM_EXTRA_BODY", value)
    with pytest.raises((SettingsError, ValidationError)):
        settings()


def test_extra_body_preserves_nested_json_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LLM_EXTRA_BODY",
        '{"options":[null,true,42,1.5,"value",{"thinking":{"type":"disabled"}}]}',
    )
    assert settings().llm_extra_body == {
        "options": [None, True, 42, 1.5, "value", {"thinking": {"type": "disabled"}}]
    }
