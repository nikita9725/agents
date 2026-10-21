"""DBOS: постоянная очередь PostgreSQL и автоматическое восстановление при старте."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from dbos import DBOS, DBOSClient, DBOSConfig, WorkflowSerializationFormat
from langchain_core.runnables import RunnableConfig

from etl_agent.domain import AgentState, RunInput
from etl_agent.infrastructure.graph import build_graph
from etl_agent.infrastructure.planner import OpenAIPlanner
from etl_agent.infrastructure.task_executor import DBOSTasks, WorkflowClient
from etl_agent.infrastructure.tools import create_tools
from etl_agent.observability import configure_logging
from etl_agent.runtime import (
    APPLICATION_NAME,
    QUEUE_NAME,
    WORKFLOW_NAME,
    RuntimeSettings,
    checkpoints,
)
from etl_agent.settings import Settings
from etl_agent.tasks import TaskService

logger = logging.getLogger(__name__)


@DBOS.step(preemptible=True)
async def _run_graph(task_id: str, goal: str, max_steps: int, request_id: str | None) -> str:
    settings = Settings()
    async with checkpoints(RuntimeSettings()) as saver:
        tools = create_tools(settings.tool_files_root)
        graph = build_graph(
            OpenAIPlanner(
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                extra_body=settings.llm_extra_body,
                tools=tools,
            ),
            tools=tools,
            checkpointer=saver,
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": task_id},
            "metadata": {"task_id": task_id, "request_id": request_id},
            "run_id": UUID(task_id),
            "recursion_limit": max(25, max_steps * 2 + 5),
        }
        snapshot = await graph.aget_state(config)
        if snapshot.values and not snapshot.next:
            return AgentState.model_validate(snapshot.values).model_dump_json()
        state = None if snapshot.values else RunInput(goal=goal, max_steps=max_steps)
        result = await graph.ainvoke(state, config, durability="sync")
        return AgentState.model_validate(result).model_dump_json()


@DBOS.workflow(
    name=WORKFLOW_NAME,
    max_recovery_attempts=3,
    serialization_type=WorkflowSerializationFormat.PORTABLE,
)
async def execute_task(task_id: str, goal: str, max_steps: int, request_id: str | None) -> str:
    return await _run_graph(task_id, goal, max_steps, request_id)


class RuntimeBackend(Protocol):
    @property
    def runtime_settings(self) -> RuntimeSettings: ...

    async def prepare(self) -> None: ...

    def launch(self) -> None: ...

    def register_queue(self) -> None: ...

    def open_client(self) -> WorkflowClient: ...

    def shutdown(self, *, workflow_completion_timeout_sec: int) -> None: ...


@dataclass(slots=True)
class DBOSRuntimeBackend:
    settings: Settings
    runtime_settings: RuntimeSettings

    async def prepare(self) -> None:
        configure_logging([self.settings.llm_api_key.get_secret_value()])
        async with checkpoints(self.runtime_settings) as saver:
            await saver.setup()
        config: DBOSConfig = {
            "name": APPLICATION_NAME,
            "system_database_url": self.runtime_settings.dbos_url,
            "executor_id": "etl-worker",
            "application_version": "etl-task-v1",
            "run_admin_server": False,
            "enable_otlp": False,
        }
        DBOS(config=config)

    def launch(self) -> None:
        DBOS.launch()

    def register_queue(self) -> None:
        DBOS.register_queue(QUEUE_NAME, worker_concurrency=5, global_concurrency=5)

    def open_client(self) -> WorkflowClient:
        return DBOSClient(
            system_database_url=self.runtime_settings.dbos_url,
            application_name=APPLICATION_NAME,
            retry_connection_errors=False,
        )

    def shutdown(self, *, workflow_completion_timeout_sec: int) -> None:
        DBOS.destroy(workflow_completion_timeout_sec=workflow_completion_timeout_sec)


@asynccontextmanager
async def task_runtime(backend: RuntimeBackend | None = None) -> AsyncIterator[TaskService]:
    driver = backend if backend is not None else DBOSRuntimeBackend(Settings(), RuntimeSettings())
    await driver.prepare()
    client: WorkflowClient | None = None
    try:
        await asyncio.to_thread(driver.launch)
        await asyncio.to_thread(driver.register_queue)
        client = await asyncio.to_thread(driver.open_client)
        logger.info("event=runtime.ready executor_id=etl-worker")
        yield TaskService(DBOSTasks(client, driver.runtime_settings))
    finally:
        try:
            await asyncio.to_thread(driver.shutdown, workflow_completion_timeout_sec=25)
        finally:
            if client is not None:
                await asyncio.to_thread(client.destroy)
