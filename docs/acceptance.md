# Приёмка дня 7

Дата проверки: 2026-10-23. Use case: диагностика локальных ETL-файлов.
Single-agent workflow, существующий FastAPI/CLI-клиент; публичный API не изменён.

## Соответствие требованиям

| Требование | Реализация и доказательство |
|---|---|
| Принимает задачу | POST /tasks, task_id, пример SSE-клиента |
| Строит plan | LangChain structured ExecutionPlan, planning один раз на задачу |
| Вызывает 2–3 tools | calculator, read_local_file, text_search; реальные integration/demo |
| Graph execution | LangGraph input → planning → acting → tools → observation → finalization |
| Working memory | AgentState: observations, notes, done_criteria, tool_results; PostgreSQL checkpoints |
| Done criteria | Evidence по успешным tool results; stop_reason в state/history |
| Guardrails | Бюджет шагов, кеш, no_progress, ошибки tools, retries/timeouts |
| Final answer и trace | final_answer/next_action и полные messages/history, SSE |
| LangSmith | Настроенный SDK и тесты дерева spans через транспорт в памяти |
| Пять demo | examples/goals.json, docs/demo.md, итоговый demo-results.json |
| Два полных trace | examples/traces/demo-4 и demo-5: JSON + SSE |
| Scope и ограничения | README: доступ к файлам, read-only, LLM, single-agent, частичный успех |

## Проверки

| Проверка | Результат |
|---|---|
| Полный `make test` после исправления | 204 passed (191 unit, 6 integration, 7 e2e) |
| `make lint` | Ruff check/format и mypy прошли |
| `make pre-commit` | Все hooks прошли |
| `uv lock --check` | Lockfile согласован |
| Пять реальных demo | 5/5 success; факты и инструменты проверены |
| Полные traces demo 4/5 | JSON и SSE сохранены; terminal snapshot совпадает с GET |
| Запуск по README | make up, /ready и пример CLI с --url на тестовом порту |

Исходный набор до изменений: 200 passed. Новый сценарий частичного результата
сначала падал из-за неверной evidence; причина исправлена, а не скрыта повтором теста.
Идентификаторы и ответы пяти запусков: [demo-results.json](../examples/traces/demo-results.json).
Подробный разбор и исходные traces: [examples/traces](../examples/traces/README.md).

Проверка использует существующие действующие LLM-настройки, настоящий провайдер
`deepseek-flash`, Docker Compose и отдельный проект `etl-agent-test` с портами
18123/15432. Рабочий volume и локальный .env не заменялись. Сам запуск не
эквивалентен проверке установки на полностью чистой машине.

## Исправление, найденное при приёмке

Смешанный пакет успешного расчёта и неуспешного чтения выявил, что модель может
помещать ошибочный tool result в evidence критерия. Prompt и одна LLM-корректировка
не обеспечили устойчивости: отдельный demo снова завершился ошибкой после
успешного прогона 203 тестов (task_id=7667b098-c12c-4e71-8854-a2be8e9b40ad).

Добавлено детерминированное правило: известные ошибочные tool results удаляются
из evidence критерия, а критерий принудительно остаётся невыполненным. Ошибки
сохраняются в observations. Неизвестные ссылки и изменение описаний критериев
по-прежнему отклоняются. Для остальных нарушений предусмотрена одна корректировка
с диагностикой валидатора через внутреннее поле MemoryInput.validation_feedback;
повторное нарушение завершает задачу ошибкой. Tools не повторяются.
Тесты проверяют демотирование критерия, сохранение наблюдений об ошибках,
неизменность исходной памяти, границу корректировок и отсутствие повторного tool.

## Граница проверки LangSmith

По выбору пользователя реальный LangSmith не подключался:
`LANGSMITH_TRACING=false`, облачная доставка **не проверена**.
Настоящий SDK проверяется unit-тестами через транспорт в памяти; реальные demo
сохраняют локальную history/SSE. Локальные артефакты не являются облачными spans
LangSmith. Self-hosted LangSmith требует Enterprise-лицензию и в стек не включён.

Инструкция включения и проверки сервиса сохранена в docs/observability.md.
Таким образом, требования к graph-based mini-product, локальным traces и
интеграции SDK проверяются отдельно от доставки в сервис LangSmith.
