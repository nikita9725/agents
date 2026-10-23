"""Проверка конфигурации локального приложения перед запуском."""

import sys

from pydantic import ValidationError

from etl_agent.settings import Settings

PLACEHOLDERS = {"", "your-key", "replace-me", "your-langsmith-api-key"}


def validate_startup(settings: Settings) -> None:
    if settings.langsmith_tracing:
        key = settings.langsmith_api_key
        if key is None or key.get_secret_value().strip() in PLACEHOLDERS:
            raise ValueError("LANGSMITH_API_KEY must be configured when tracing is enabled")
        if not settings.langsmith_project.strip():
            raise ValueError("LANGSMITH_PROJECT must be nonempty when tracing is enabled")
    if not settings.llm_model.strip() or settings.llm_model == "your-model-name":
        raise ValueError("LLM_MODEL must be configured for real task execution")
    if settings.llm_api_key.get_secret_value().strip() in PLACEHOLDERS:
        raise ValueError("LLM_API_KEY must be configured for real task execution")


def main() -> None:
    try:
        validate_startup(Settings())
    except ValueError as exc:
        # Pydantic errors may include rejected input values, including credentials.
        if isinstance(exc, ValidationError):
            print(
                exc.errors(include_input=False, include_context=False, include_url=False),
                file=sys.stderr,
            )
        else:
            print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
