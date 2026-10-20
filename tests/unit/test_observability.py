import io
import logging

import pytest

from etl_agent.domain import NextAction
from etl_agent.exceptions import PlanningError
from etl_agent.observability import DiagnosticFormatter, configure_logging

pytestmark = pytest.mark.unit


def test_formatter_redacts_secrets_in_chained_tracebacks() -> None:
    secret = "unique-secret-api-key"
    configure_logging([secret])
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(DiagnosticFormatter("%(levelname)s %(message)s"))
    logger = logging.getLogger("test.safe_diagnostics")
    logger.addHandler(handler)
    try:
        try:
            try:
                raise RuntimeError(f"Bearer {secret} postgresql://user:password@host/db")
            except RuntimeError as exc:
                raise PlanningError("provider failed") from exc
        except PlanningError:
            logger.exception("event=test.failed")
    finally:
        logger.removeHandler(handler)
    text = output.getvalue()
    assert secret not in text
    assert "user:password" not in text
    assert "Traceback" in text
    assert "RuntimeError" in text
    assert "PlanningError: provider failed" in text


def test_model_validation_traceback_omits_response_content() -> None:
    from pydantic import ValidationError

    invalid_response = {"action": "private model response"}
    try:
        NextAction.model_validate(invalid_response)
    except ValidationError as exc:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "failed", (), (type(exc), exc, exc.__traceback__)
        )
    text = DiagnosticFormatter("%(message)s").format(record)
    assert "private model response" not in text
    assert "ValidationError" in text
    assert "rationale" in text


def test_sdk_http_error_body_is_omitted_from_traceback() -> None:
    import httpx

    try:
        raise httpx.HTTPStatusError(
            "private-upstream-body-with-unknown-key",
            request=httpx.Request("GET", "http://executor"),
            response=httpx.Response(500),
        )
    except httpx.HTTPStatusError as exc:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "failed", (), (type(exc), exc, exc.__traceback__)
        )
    text = DiagnosticFormatter("%(message)s").format(record)
    assert "private-upstream-body-with-unknown-key" not in text
    assert "HTTPStatusError" in text and "Traceback" in text
