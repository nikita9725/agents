"""Единственный публичный HTTP API приложения."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import RequestResponseEndpoint

from etl_agent.domain import Model, RunInput
from etl_agent.observability import request_id
from etl_agent.tasks import ExecutorUnavailable, Task, TaskError, TaskNotFound, TaskService
from etl_agent.worker import task_runtime

logger = logging.getLogger(__name__)


class InputIssue(Model):
    loc: list[str | int]
    type: str


class ApiError(Model):
    error: TaskError
    details: list[InputIssue] | None = None


def _error_response(
    status: int, code: TaskError, details: list[InputIssue] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ApiError(error=code, details=details).model_dump(mode="json", exclude_none=True),
    )


@dataclass(slots=True)
class AppResources:
    service: TaskService | None = None


def _get_resources(application: FastAPI) -> AppResources:
    resources: object = getattr(application.state, "resources", None)
    if not isinstance(resources, AppResources):
        raise ExecutorUnavailable
    return resources


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    resources = _get_resources(application)
    if resources.service is not None:
        yield
        return
    async with task_runtime() as service:
        resources.service = service
        try:
            yield
        finally:
            resources.service = None


def get_service(request: Request) -> TaskService:
    service = _get_resources(request.app).service
    if service is None:
        raise ExecutorUnavailable
    return service


async def _stream_events(tasks: TaskService, initial: Task) -> AsyncIterator[str]:
    async for event in tasks.events(initial):
        yield event.encode()


async def not_found(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(404, TaskError(code="not_found", message="Task not found"))


async def unavailable(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(
        503, TaskError(code="executor_unavailable", message="Task executor is unavailable")
    )


async def invalid(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise TypeError("Validation handler requires RequestValidationError")
    return _error_response(
        422,
        TaskError(code="invalid_input", message="Invalid request"),
        [InputIssue(loc=list(error["loc"]), type=error["type"]) for error in exc.errors()],
    )


router = APIRouter()


async def diagnostics(request: Request, call_next: RequestResponseEndpoint) -> Response:
    try:
        correlation_id = str(UUID(request.headers.get("X-Request-ID", "")))
    except ValueError:
        correlation_id = str(uuid4())
    token = request_id.set(correlation_id)
    started = perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("event=request.failed")
            response = _error_response(
                500, TaskError(code="internal_error", message="Internal server error")
            )
        response.headers["X-Request-ID"] = correlation_id
        logger.info(
            "event=request.completed method=%s status=%s elapsed_ms=%.1f task_id=%s",
            request.method,
            response.status_code,
            (perf_counter() - started) * 1000,
            request.path_params.get("task_id", "-"),
        )
        return response
    finally:
        request_id.reset(token)


@router.get("/ok")
async def ok() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(tasks: Annotated[TaskService, Depends(get_service)]) -> dict[str, str]:
    await tasks.ready()
    return {"status": "ready"}


@router.post("/tasks", response_model=Task, status_code=202)
async def create_task(
    body: RunInput,
    response: Response,
    tasks: Annotated[TaskService, Depends(get_service)],
) -> Task:
    task = await tasks.create(body)
    response.headers["Location"] = f"/tasks/{task.task_id}"
    return task


@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: UUID, tasks: Annotated[TaskService, Depends(get_service)]) -> Task:
    return await tasks.get(task_id)


@router.post("/tasks/{task_id}/cancel", response_model=Task)
async def cancel_task(task_id: UUID, tasks: Annotated[TaskService, Depends(get_service)]) -> Task:
    return await tasks.cancel(task_id)


@router.get(
    "/tasks/{task_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE progress, task snapshots and safe stream errors",
        }
    },
)
async def events(
    task_id: UUID, tasks: Annotated[TaskService, Depends(get_service)]
) -> StreamingResponse:
    initial = await tasks.get(task_id)

    return StreamingResponse(
        _stream_events(tasks, initial),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(service: TaskService | None = None) -> FastAPI:
    application = FastAPI(
        title="ETL Tasks API",
        lifespan=lifespan,
        responses={code: {"model": ApiError} for code in (404, 422, 500, 503)},
    )
    application.state.resources = AppResources(service=service)
    application.include_router(router)
    application.middleware("http")(diagnostics)
    application.add_exception_handler(TaskNotFound, not_found)
    application.add_exception_handler(ExecutorUnavailable, unavailable)
    application.add_exception_handler(RequestValidationError, invalid)
    return application


app = create_app()
