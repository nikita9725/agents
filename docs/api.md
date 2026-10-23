# API задач

Единственный публичный API приложения — FastAPI на `http://127.0.0.1:8123`.
Спецификация: `/openapi.json`, Swagger: `/docs`. Клиент использует только `task_id`.

## Создание и получение

```sh
curl -X POST http://127.0.0.1:8123/tasks \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Нет sales.csv","max_steps":5}'
```

`goal` — непустой текст; `max_steps` — положительное целое, по умолчанию 5.
Бюджет ограничивает успешные вызовы acting, а не число отдельных tools.
Последний набор tools выполняется, после чего агент формирует ответ с отметкой
об исчерпании бюджета. Начальное планирование, обновление памяти и финализация в этот бюджет не входят.
Если на последнем шаге выполнены критерии, причина завершения — criteria_met.
Лишние поля запрещены. Ответ `202 Accepted` и `Location: /tasks/{task_id}`
означают, что фоновая работа создана; ответ модели ещё не требуется.

```json
{"task_id":"46d34898-1d45-4bc4-a2fd-0682984b60dd","status":"pending","result":null,"error":null}
```

`GET /tasks/{task_id}` возвращает ту же структуру. Статусы:

| Статус | Значение |
|---|---|
| pending | Задача принята, ожидает исполнения |
| running | Выполняется |
| success | Успешно завершена; `result` содержит цель, историю, шаг, действие и ответ |
| error | Выполнение завершилось ошибкой; `error.code=execution_failed` |
| timeout | Превышен лимит исполнения; `error.code=timeout` |
| interrupted | Выполнение прервано |

Ошибки выполнения читаются через GET с HTTP 200. HTTP 202 при создании не
гарантирует успешный результат. Проверяйте `status`, а не наличие фрагмента ответа.

`POST /tasks/{task_id}/cancel` прерывает задачу и возвращает её актуальное описание.
Повторная отмена или отмена завершённой задачи не создаёт новую работу.

## События

`GET /tasks/{task_id}/events` — SSE, `Content-Type: text/event-stream`.

```text
event: progress
data: {"task_id":"46d34898-1d45-4bc4-a2fd-0682984b60dd","sequence":2,"node":"planning","role":"assistant","content":"План: прочитать лог, затем вычислить процент"}

event: task
data: {"task_id":"46d34898-1d45-4bc4-a2fd-0682984b60dd","status":"success","result":{},"error":null}

```

В примере `result` сокращён; его реальная схема совпадает с GET задачи.
`progress.node`: `input`, `planning`, `acting`, `tools`, `observation`, `finalization`. `role` и `content` совпадают с записью
`result.history`. `sequence` — порядковый номер записи начиная с 1.
События строятся по сохранённой истории с опросом раз в 200 мс: это завершённые
записи trace, не токены LLM. Tools сохраняет результат в checkpoint, observation
добавляет его в историю. Повторные acting/observation передаются отдельно.
`task` содержит снимки при смене статуса и итоговый результат. Для завершённой
задачи сохранённый trace и итог отдаются сразу. При переподключении история
воспроизводится с начала; дедупликация по `(task_id, sequence)`.
Отключение SSE не отменяет задачу.

Ошибка исполнителя до открытия потока возвращает HTTP 503. После открытия —
`event: error` с `task_id` и безопасной ошибкой `executor_unavailable`; клиент может
подключиться заново. Это ошибка наблюдения, а не новый статус задачи.

## Ошибки и диагностика

- `404`: задача не найдена.
- `422`: неверный ввод, лишние поля или некорректный UUID.
- `503`: исполнитель недоступен либо его ответ не позволяет получить задачу.
- `500`: непредвиденная ошибка API.

HTTP-ошибки содержат объект `error` с `code` и `message`; 422 также содержит
местоположения и типы ошибок без отклонённых значений. Ответы не содержат
traceback, внутренние идентификаторы или сырые ошибки провайдера.
`X-Request-ID` помогает найти серверные логи.

`GET /ok` проверяет процесс FastAPI. `GET /ready` проверяет доступность исполнителя,
но не выполняет LLM-запрос. Порт приложения публикуется только на loopback.

В `result.messages` сохраняются типизированные сообщения LangChain: human, ai и tool.
AIMessage содержит tool_calls с именем, аргументами и id; ToolMessage — observation,
status и соответствующий tool_call_id. `result.history` допускает роль `tool`.
`result.next_action` сохраняет action/rationale, `result.final_answer` содержит ответ
по наблюдениям. `result.plan.steps` содержит начальный краткий план; `pending_observations` пуст
после успешного завершения. Trace `planning → acting → observation → acting`
сохраняется в истории и передаётся через SSE без удаления повторных узлов.

## Рабочая память и изменения состояния

`result.observations` содержит краткие наблюдения `{summary, evidence}`;
`result.notes` — полезную выжимку; `result.done_criteria` — критерии
`{description, satisfied, evidence}`. Evidence ссылается на `result.tool_results[].id`.
Каждый ToolResult содержит `id`, `name`, `args`, `message` (исходный ToolMessage).
Повторные вызовы используют сохранённый результат, но получают свой tool_call_id.
`result.last_batch_new_results` показывает число новых результатов последнего пакета.
`result.stop_reason`: criteria_met, no_new_results, step_limit или actor_finished.

`HistoryEntry` и SSE `progress` содержат необязательный `state_delta` (null у
старых записей). Последняя запись каждого узла содержит его изменения state:
например `{"notes":["expected_rows=120, rejected_rows=30"],"pending_observations":0}`.
Tools сообщает `new_result_ids` и `reused_result_ids`. Полные файлы и архив переписки
в delta не копируются. Клиент печатает delta рядом с текстом узла.

Память и результаты сохраняются в PostgreSQL checkpoints, итоговый state — в DBOS.
Новые поля старых завершённых результатов по умолчанию пусты. Ограничения памяти
относятся к контексту LLM; полный архив результатов остаётся доступен в result.
