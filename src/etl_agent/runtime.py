"""Настройки постоянной очереди и checkpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

APPLICATION_NAME = "etl-agent"
WORKFLOW_NAME = "etl_task"
QUEUE_NAME = "etl_tasks"


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    task_database_url: SecretStr = SecretStr(
        "postgresql://postgres:postgres@127.0.0.1:5432/etl_agent"
    )

    @property
    def dbos_url(self) -> str:
        return self.task_database_url.get_secret_value().replace(
            "postgresql://", "postgresql+psycopg://", 1
        )

    @property
    def postgres_url(self) -> str:
        return self.task_database_url.get_secret_value().replace(
            "postgresql+psycopg://", "postgresql://", 1
        )


@asynccontextmanager
async def checkpoints(settings: RuntimeSettings) -> AsyncIterator[AsyncPostgresSaver]:
    allowed = [
        ("etl_agent.domain", "HistoryEntry"),
        ("etl_agent.domain", "NextAction"),
        ("etl_agent.domain", "RunInput"),
    ]
    serde = JsonPlusSerializer(allowed_json_modules=allowed, allowed_msgpack_modules=allowed)
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_url, serde=serde) as saver:
        yield saver
