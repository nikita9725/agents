# Агент диагностики ETL

Единственный публичный интерфейс — FastAPI API задач. LangGraph вызывает инструменты через настоящую LLM, анализирует observations
и предлагает следующий шаг диагностики. Очередь, статусы и восстановление
обеспечивает открытая библиотека DBOS с хранением в PostgreSQL.
LangSmith и лицензия Agent Server для запуска не нужны.

```mermaid
flowchart LR
    subgraph app[Один процесс приложения]
        api[FastAPI]
        executor[DBOS executor]
        graph[LangGraph: input → planning ⇄ tools → finalization]
    end
    client[Клиент: task_id] --> api
    api --> db[(PostgreSQL: DBOS очередь и результаты)]
    db --> executor
    executor --> graph
    graph --> llm[LLM]
    graph --> checkpoints[(PostgreSQL: checkpoints)]
    client -->|GET / SSE| api
```

Агент читает локальные UTF-8-файлы, ищет текст и выполняет арифметические расчёты.
Три инструмента оформлены через LangChain `@tool` и Pydantic. Файлы доступны только
внутри `TOOL_FILES_ROOT` (по умолчанию `examples/tool_data`); изменения данных агент не выполняет.
[Инструменты и примеры дня 2](docs/planning.md).

## Запуск

Нужны Docker, Docker Compose v2, uv и Python 3.13.

```sh
cp .env.example .env
make setup
```

Настройте `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` в `.env`.
Для DeepSeek:

```dotenv
LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}
```

```sh
make infra # PostgreSQL в Docker, с ожиданием готовности
make run   # Python на хосте, FastAPI и DBOS в одном процессе
# или весь стек в Docker:
make up    # запуск в фоне с ожиданием готовности
```

Пользовательские команды запуска определены в Makefile. `make infra` поднимает PostgreSQL;
lifespan приложения автоматически инициализирует схемы DBOS и LangGraph при старте.
FastAPI доступен на http://127.0.0.1:8123 (`AGENT_PORT`).
Swagger: `/docs`, OpenAPI: `/openapi.json`, health: `/ok`, готовность БД: `/ready`.
DBOS работает внутри процесса FastAPI. PostgreSQL публикуется только на localhost:
`127.0.0.1:${POSTGRES_PORT:-5432}`. Локальный процесс читает `TASK_DATABASE_URL`
из .env; контейнер API использует `postgres:5432`. При смене POSTGRES_PORT обновите
порт в TASK_DATABASE_URL.

Перед переходом с `make up` на `make run` выполните `make down`, затем
`make infra`: оба приложения используют один порт и один DBOS executor_id,
поэтому их нельзя запускать одновременно с общей БД. Данные сохраняются.
Ctrl+C останавливает локальный API и исполнитель, PostgreSQL продолжает работать.
Локальные логи выводятся в терминал; `make logs` показывает логи Docker.

## Создание задачи

```sh
uv run --locked python examples/client.py 'Не загрузились продажи: нет sales.csv'
```

`POST /tasks` сохраняет задачу в постоянной очереди и возвращает `202` с `task_id`.
`GET /tasks/{task_id}` возвращает статус и результат; `/cancel` прерывает задачу,
`/events` отдаёт SSE завершённых узлов и итог. Клиент не работает с внутренними
идентификаторами исполнения. [Контракт API](docs/api.md).

## Надёжность

DBOS сохраняет очередь, статус, дедлайн и результат в PostgreSQL. Исполнитель внутри процесса FastAPI
обрабатывает до пяти задач одновременно. Docker автоматически перезапускает
процесс после аварии в режиме `make up`. При локальном запуске повторите
`make run` вручную. DBOS при старте восстанавливает незавершённые задачи своего
стабильного executor_id. Сохранённые LangGraph checkpoints позволяют продолжить
с незавершённого узла, сохранив task_id. Ручной POST или resume не требуется.

Planning и finalization: до трёх попыток временных ошибок, по 30 секунд на попытку. Ошибки
400/401/403 не повторяются. DBOS ограничивает выполнение задачи 180 секундами
после получения из очереди; отмена прерывает асинхронный шаг.
Бюджет `max_steps` по умолчанию — 5 вызовов planning; финализация выполняется отдельно.
Если процесс умер до сохранения результата узла, LLM или read-only tools могут быть вызваны повторно.
Перед обновлением графа завершите или отмените старые активные задачи; готовые результаты сохраняются.

## Проверки и остановка

```sh
make unit-test         # без Docker, ключей и сети
make integration-test  # PostgreSQL, настоящее приложение и LLM
make e2e-test          # SSE, отмена, очередь, SIGKILL и восстановление
make test              # весь набор, без подмен в integration/e2e
make lint
make pre-commit
make logs
make down              # volumes сохраняются
```

Тестовый проект `etl-agent-test` имеет отдельный volume и порт 18123
(`AGENT_TEST_PORT`), PostgreSQL — порт 15432 (`POSTGRES_TEST_PORT`). Наборы integration/e2e нельзя запускать одновременно.
Текущие данные находятся в volume `etl-agent_task_data`. Старые volumes Agent
Server и `agents_postgres_data` сохраняются; автоматической миграции задач из
лицензируемого runtime нет. [Документация](docs/README.md).
