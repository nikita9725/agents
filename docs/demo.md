# Пять demo-задач

Запустите приложение по [README](../README.md). Используются неизменённые файлы
`examples/tool_data`, стандартный `max_steps=5` и настоящий LLM.

| № | Цель | Проверяемый результат |
|---|---|---|
| 1 | Арифметика | calculator → 25% |
| 2 | Чтение лога | read_local_file → rejected_rows=30 |
| 3 | Поиск ошибки | text_search → 2026-99-21, строка 3 |
| 4 | Полная диагностика | Чтение, поиск и расчёт; 25% и ошибочная дата |
| 5 | Частичный результат | calculator → 4; missing.log → error; ограничение в ответе |

## Demo 1

```sh
uv run --locked python examples/client.py 'Вычисли (120-90)/120*100 инструментом calculator. Укажи процент отклонённых строк.'
```

## Demo 2

```sh
uv run --locked python examples/client.py 'Прочитай etl.log инструментом read_local_file. Сколько строк отклонено?'
```

## Demo 3

```sh
uv run --locked python examples/client.py 'Найди invalid date в etl.log инструментом text_search. Укажи ошибочное значение даты.'
```

## Demo 4

```sh
uv run --locked python examples/client.py 'Сначала прочитай etl.log через read_local_file. Затем найди invalid date через text_search и вычисли calculator процент rejected_rows от expected_rows из прочитанного лога. Укажи процент и ошибочную дату.'
```

## Demo 5

```sh
uv run --locked python examples/client.py 'Вычисли 2+2 через calculator и прочитай missing.log через read_local_file. Если файл недоступен, явно укажи ограничение и не придумывай его содержимое.'
```

## Как оценить запуск

У каждого запуска должны быть план, tool calls, observations, рабочая память,
финализация и причина остановки. В demo 4 результат чтения должен предшествовать
расчёту. В demo 5 ошибка чтения не должна уничтожить успешный расчёт;
агент должен честно указать недоступность файла. Технический success не означает
полного выполнения цели.

Точный текст ответа, формула расчёта и число дополнительных шагов могут
отличаться. Проверяйте факты и связи tool_call_id → ToolMessage → evidence.

Полная история остаётся в `GET /tasks/{task_id}` и доступна повторно через SSE:

```sh
curl --fail -N 'http://127.0.0.1:8123/tasks/<task_id>/events'
curl --fail 'http://127.0.0.1:8123/tasks/<task_id>'
```

[Два сохранённых реальных trace](../examples/traces/README.md).
