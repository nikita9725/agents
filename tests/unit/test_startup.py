from unittest.mock import patch

import pytest
from pydantic import SecretStr

from etl_agent.settings import Settings
from etl_agent.startup import validate_startup

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("key", [None, "", "your-key", "replace-me", "your-langsmith-api-key"])
def test_langsmith_is_not_required(key: str | None) -> None:
    settings = Settings(
        _env_file=None,
        llm_model="test-model",
        llm_api_key=SecretStr("test-key"),
        langsmith_api_key=SecretStr(key) if key is not None else None,
    )
    with (
        patch("etl_agent.startup.Settings", return_value=settings),
    ):
        validate_startup()


@pytest.mark.parametrize(
    ("model", "llm_key", "error"),
    [
        ("test-model", "test-key", None),
        ("your-model-name", "test-key", "LLM_MODEL must be configured"),
        (" ", "test-key", "LLM_MODEL must be configured"),
        ("test-model", "replace-me", "LLM_API_KEY must be configured"),
        ("test-model", " ", "LLM_API_KEY must be configured"),
    ],
)
def test_provider_configuration(model: str, llm_key: str, error: str | None) -> None:
    settings = Settings(
        _env_file=None,
        llm_model=model,
        llm_api_key=SecretStr(llm_key),
        langsmith_api_key=SecretStr("unit-key"),
    )
    with patch("etl_agent.startup.Settings", return_value=settings):
        if error is None:
            validate_startup()
        else:
            with pytest.raises(ValueError, match=error):
                validate_startup()
