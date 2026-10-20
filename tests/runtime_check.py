"""Инфраструктурные проверки постоянной очереди из Docker-сети."""

import asyncio
import sys
from uuid import UUID

from dbos import DBOSClient, EnqueueOptions, WorkflowSerializationFormat
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from etl_agent.domain import HistoryEntry, NextAction
from etl_agent.runtime import (
    APPLICATION_NAME,
    QUEUE_NAME,
    WORKFLOW_NAME,
    RuntimeSettings,
    checkpoints,
)


class RuntimeInfo(BaseModel):
    run_id: str
    status: str
    run_count: int
    has_input: bool
    has_action: bool
    history_count: int


class Snapshot(BaseModel):
    history: list[HistoryEntry] = Field(default_factory=list)
    next_action: NextAction | None = None


async def inspect(task_id: str, action: str) -> RuntimeInfo:
    settings = RuntimeSettings()
    client = await asyncio.to_thread(
        DBOSClient, system_database_url=settings.dbos_url, application_name=APPLICATION_NAME
    )
    try:
        if action == "defer":
            options: EnqueueOptions = {
                "workflow_name": WORKFLOW_NAME,
                "queue_name": QUEUE_NAME,
                "workflow_id": task_id,
                "delay_seconds": 30,
                "workflow_timeout": 180,
                "serialization_type": WorkflowSerializationFormat.PORTABLE,
            }
            await client.enqueue_async(
                options, task_id, "Учебная задача: файл sales.csv отсутствует", 1, None
            )
        workflows = await client.list_workflows_async(workflow_ids=[task_id], name=WORKFLOW_NAME)
        assert len(workflows) == 1
        workflow = workflows[0]
        config: RunnableConfig = {"configurable": {"thread_id": task_id}}
        async with checkpoints(settings) as saver:
            saved = await saver.aget_tuple(config)
            snapshot = Snapshot.model_validate(saved.checkpoint["channel_values"] if saved else {})
            history = [item async for item in saver.alist(config)]
        return RuntimeInfo(
            run_id=workflow.workflow_id,
            status="running" if workflow.status == "PENDING" else workflow.status.lower(),
            run_count=len(workflows),
            has_input=bool(snapshot.history),
            has_action=snapshot.next_action is not None,
            history_count=len(history),
        )
    finally:
        await asyncio.to_thread(client.destroy)


if __name__ == "__main__":
    print(asyncio.run(inspect(str(UUID(sys.argv[1])), sys.argv[2])).model_dump_json())
