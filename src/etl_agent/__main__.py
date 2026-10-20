"""Локальный запуск единого процесса FastAPI и DBOS."""

import uvicorn

from etl_agent.settings import Settings
from etl_agent.startup import main as check_config


def main() -> None:
    check_config()
    uvicorn.run(
        "etl_agent.api:app",
        host="127.0.0.1",
        port=Settings().agent_port,
        workers=1,
        reload=False,
        access_log=False,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
