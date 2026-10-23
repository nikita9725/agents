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

Финальные результаты полного прогона и demo будут записаны после завершения
проверок. Исходный набор до изменений: 200 passed; после добавления ограниченного
исправления памяти unit-набор: 190 passed.

Проверка использует существующие действующие LLM-настройки, настоящий провайдер
`deepseek-flash`, Docker Compose и отдельный проект `etl-agent-test` с портами
18123/15432. Рабочий volume и локальный .env не заменялись. Сам запуск не
эквивалентен проверке установки на полностью чистой машине.

## Исправление, найденное при приёмке

Смешанный пакет успешного расчёта и неуспешного чтения выявил, что модель может
помещать ошибочный tool result в evidence критерия. Одного уточнения prompt было
недостаточно. Теперь observation разрешает одну корректировку с диагностикой
валидатора через внутреннее поле MemoryInput.validation_feedback; tools не
повторяются. Повторное нарушение отклоняется. Pydantic, неизменность критериев
и проверка evidence сохранены; тесты проверяют обе ветки и границу попыток.

## Граница проверки LangSmith

По выбору пользователя реальный LangSmith не подключался:
`LANGSMITH_TRACING=false`, облачная доставка **не проверена**.
Настоящий SDK проверяется unit-тестами через транспорт в памяти; реальные demo
сохраняют локальную history/SSE. Локальные артефакты не являются облачными spans
LangSmith. Self-hosted LangSmith требует Enterprise-лицензию и в стек не включён.

Инструкция включения и проверки сервиса сохранена в docs/observability.md.
Таким образом, требования к graph-based mini-product, локальным traces и
интеграции SDK проверяются отдельно от доставки в сервис LangSmith.
