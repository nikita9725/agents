# Архитектура

- FastAPI — единственный публичный API задач, ошибки, request_id и SSE.
- TaskService — типизированный асинхронный контракт приложения.
- DBOSTasks — адаптер очереди и статусов DBOS, чтение checkpoints.
- DBOS worker — постоянная очередь PostgreSQL, конкурентность и восстановление.
- LangGraph — граф input → planning → acting → tools → observation → acting/finalization, native retry/timeout policies.
- OpenAIPlanner — настоящий провайдер, структурированный план, tool calling, structured finalization и классификация ошибок.

## Постоянное исполнение

Одна задача имеет один DBOS workflow_id, совпадающий с task_id. Фоновая работа
подтверждается клиенту только после записи в PostgreSQL. HTTP-обработчик передаёт работу DBOS и
не хранит соответствия в памяти и не запускает собственные watchdog или очередь.
DBOSClient работает с БД через асинхронные методы, блокирующий код изолирует в потоках.

Lifespan регистрирует workflow etl_task и очередь etl_tasks с concurrency=5.
Внутри workflow выполняется preemptible async DBOS step с LangGraph. Checkpoints
сохраняются синхронно после узлов через AsyncPostgresSaver. При повторном исполнении
шага граф читает snapshot и продолжает незавершённый узел; готовый ответ повторно
не вычисляется. Отмена DBOS прерывает async step и вложенный вызов модели.

В Compose один процесс FastAPI с DBOS со стабильным executor_id=etl-worker и application_version=
etl-task-v1. При рестарте DBOS автоматически восстанавливает его pending workflows;
не более трёх recovery attempts. Настройки executor_id предназначены для одного
процесса приложения, их нельзя копировать без изменения на несколько независимых процессов приложения.

## Хранение и запуск

PostgreSQL хранит схему dbos (очередь, статусы, шаги, результаты) и стандартные
таблицы checkpoints LangGraph. Инициализация выполняется библиотеками при старте приложения.
Новый volume task_data отделён от старых volumes лицензируемого Agent Server.
Старые данные не удаляются и не мигрируются автоматически.

Сервис api совмещает FastAPI и DBOS в одном процессе и использует Dockerfile Python 3.13 с uv --locked.
FastAPI и PostgreSQL публикуются только на localhost. Redis и лицензируемый Agent Server удалены из
новой конфигурации. LangSmith остаётся необязательным сервисом трассировки LLM.
Команды сборки и запуска находятся в Makefile. Аварийными сценариями
тестового стека управляет Python-помощник.

## Жизненный цикл общего процесса

Контейнер api запускает один Uvicorn worker без reload. FastAPI lifespan сначала
создаёт checkpoints, запускает DBOS и регистрирует очередь, затем предоставляет
TaskService обработчикам. Блокирующие операции запуска выполняются через
asyncio.to_thread. DBOS исполняет асинхронные задачи в своём event loop внутри
того же процесса; PostgreSQL остаётся отдельным контейнером.

При завершении API сначала перестаёт принимать запросы (до 5 секунд на HTTP),
затем lifespan скрывает TaskService, даёт DBOS до 25 секунд на завершение работы
и закрывает клиент БД. Docker предоставляет 40 секунд. При локальном запуске `make run` запускает Python на хосте, а `make infra` —
PostgreSQL в Docker. Локальный и контейнерный API не запускаются одновременно
с одной БД: executor_id предназначен для одного процесса.
При SIGKILL API временно
недоступен; после автоматического рестарта Docker (или повторного `make run`) DBOS восстанавливает задачи из БД.
Отдельного сервиса worker и административного HTTP-порта DBOS больше нет.
Имена workflow, step, очереди, executor и версия сохранены; миграция таблиц
и удаление volume не требуются.
