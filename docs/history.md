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
