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
        graph[LangGraph: planning → acting → tools → observation]
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
[Планирование, инструменты и примеры дня 4](docs/planning.md).

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
`/events` отдаёт SSE завершённых узлов и итог. Клиент печатает trace `plan → action → observation → next action` через SSE,
затем JSON результата. После каждого узла выводится `state Δ`: изменения памяти, новые/повторные результаты
и причина остановки. Повторные действия сохраняются в порядке выполнения.
Клиент использует только публичный task_id. [Контракт API](docs/api.md).

## Надёжность

DBOS сохраняет очередь, статус, дедлайн и результат в PostgreSQL. Исполнитель внутри процесса FastAPI
обрабатывает до пяти задач одновременно. Docker автоматически перезапускает
процесс после аварии в режиме `make up`. При локальном запуске повторите
`make run` вручную. DBOS при старте восстанавливает незавершённые задачи своего
стабильного executor_id. Сохранённые LangGraph checkpoints позволяют продолжить
с незавершённого узла, сохранив task_id. Ручной POST или resume не требуется.

Planning, acting, observation и finalization: до трёх попыток временных ошибок, по 30 секунд на попытку. Ошибки
400/401/403 не повторяются. DBOS ограничивает выполнение задачи 180 секундами
после получения из очереди; отмена прерывает асинхронный шаг.
Бюджет `max_steps` по умолчанию — 5 успешных вызовов acting; начальный план и финализация выполняются отдельно.
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

## Рабочая память — день 4

AgentState содержит observations, tool_results, notes и done_criteria. Observation
обновляет полезную выжимку через LLM, следующий actor использует её вместо всей
переписки. Одинаковые вызовы инструментов получают сохранённый результат; при
выполнении критериев или отсутствии новых данных граф завершает диагностику.
Память и исходные результаты сохраняются в PostgreSQL checkpoints, итоговый state —
в DBOS. Полная история остаётся для аудита. [Граф, лимиты и восстановление](docs/graph.md).


## Guardrails — день 5

Агент останавливается по критериям, решению actor, повторному пакету вызовов,
лимиту шагов или после двух пакетов подряд без нового успешного результата
(`no_progress`). Кеш предотвращает повторное исполнение, включая циклы A → B → A.
Ошибки аргументов и исключения tools сохраняются как observations со status=error;
агент может исправить вызов, а остальные tools пакета продолжают выполняться.
Причина завершения видна в ответе, trace и логах; счётчик пакетов без прогресса
сохраняется в checkpoints. [Правила остановки](docs/graph.md) и
[проверяемые сценарии](docs/testing.md#день-5).


## Наблюдаемость — день 6

Граф пишет выбранные узлы и действия, вызовы tools, observations, краткие изменения
состояния, попытки, длительности и ошибки. `task_id` связывает логи, SSE,
checkpoints и LangSmith; при восстановлении создаётся новый `run_id`.

LangSmith включается через `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT`,
`LANGSMITH_API_KEY` и при необходимости `LANGSMITH_ENDPOINT` в `.env`.
По умолчанию внешняя трассировка выключена. Ошибка доставки не меняет результат задачи.
[Настройка, три сценария, схемы выполнения и failure modes](docs/observability.md#день-6).
