"""Настройки окружения с маскированием секретов в строковом представлении."""

import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, JsonValue, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tool_files_root: Path = Path("examples/tool_data")

    agent_port: int = Field(default=8123, ge=1, le=65535)

    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str
    llm_extra_body: Annotated[dict[str, JsonValue], NoDecode] = Field(default_factory=dict)
    llm_api_key: SecretStr
    langsmith_tracing: bool = False
    langsmith_project: str = "etl-agent"
    langsmith_api_key: SecretStr | None = None

    @field_validator("llm_extra_body", mode="before")
    @classmethod
    def _parse_extra_body(cls, value: object) -> object:
        # Decode here so JSON null is validated rather than treated as a missing env value.
        if isinstance(value, str):
            decoded: object = json.loads(value)
            return decoded
        return value
