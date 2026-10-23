# История и checkpoints

Клиент получает историю и результат через GET /tasks/{task_id}. Записи истории
содержат роль, имя узла и текст. Внутренний LangGraph thread_id совпадает с task_id
и используется только сервером.

DBOS сохраняет workflow с тем же UUID в PostgreSQL. В отдельных стандартных
таблицах AsyncPostgresSaver сохраняет состояние, историю и незавершённые узлы графа.
После аварии процесса приложения DBOS восстанавливает workflow, а граф продолжает с checkpoint.
Завершённый workflow возвращает сохранённый JSON-результат без нового вызова модели.

Новые данные находятся в volume task_data. Старые volumes Agent Server и
agents_postgres_data сохранены, автоматического переноса прежних задач нет.

Состояние дня 2 содержит `messages`: HumanMessage, AIMessage с tool_calls и
ToolMessage с observations. История допускает role=tool и node=tools. У завершённых
результатов дня 1 messages по умолчанию пуст; их чтение сохраняется. Миграция
активных checkpoints старого графа не выполняется.

## agents_day03

Состояние содержит `plan.steps` и `pending_observations`. Planner формирует план один
раз. Tools сохраняет ожидающие ToolMessage; observation переносит их в messages и
историю и очищает pending_observations. Checkpoint после tools позволяет продолжить
обработку observations без повторного выполнения сохранённых инструментов.

История содержит planning (план), acting (решение и tool calls), observation
(результат каждого инструмента со status и tool_call_id), finalization (ответ).
SSE нумерует записи истории начиная с 1 и сохраняет повторные acting/observation.
Старые результаты читаются с plan=null и pending_observations=[], прежнее имя
узла tools остаётся допустимым. Перед обновлением завершите или отмените активные
задачи: незавершённые checkpoints старого графа не мигрируются.
