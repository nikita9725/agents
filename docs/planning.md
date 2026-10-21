# Планирование и инструменты — agents_day02

Агент использует настоящий OpenAI-совместимый провайдер через LangChain.
`Planner.plan(messages)` возвращает AIMessage с текстом или tool calls;
`Planner.finalize(messages)` возвращает FinalResponse с `final_answer` и
`next_action` (`action`, `rationale`). Метод `propose(goal)` сохранён как
совместимый однократный вызов первого дня; рабочий граф его не использует.

Инструменты оформлены `@tool` с Pydantic `args_schema`. Каждый успешный результат
сначала валидируется Pydantic-моделью результата и сериализуется в JSON.
LangChain оборачивает его в ToolMessage. Ошибки аргументов и ожидаемые ошибки
операций возвращаются с `status=error` и JSON-полем `error`.

| Инструмент | Вход | Результат |
|---|---|---|
| calculator | expression | result: конечное число |
| read_local_file | path | path, content, truncated |
| text_search | path, query | path, matches: line/text, truncated |

Калькулятор поддерживает числа, скобки, унарные знаки и `+ - * /` через AST,
без выполнения Python. Длина выражения — до 512 символов, дерево — до 100 узлов.
Файловые инструменты принимают только относительные пути внутри TOOL_FILES_ROOT;
проверка выполняется после разрешения symlink. Читаются только обычные UTF-8-файлы,
не более 64 КиБ. `text_search` ищет буквальную строку с учётом регистра и возвращает
до 50 совпавших строк с номерами начиная с 1. `truncated` отмечает ограничение
размера чтения или числа совпадений.

## Настройки

| Переменная | Назначение |
|---|---|
| LLM_BASE_URL | URL OpenAI-совместимого API |
| LLM_MODEL | Модель с поддержкой tool calling |
| LLM_API_KEY | Ключ из окружения или локального .env |
| LLM_EXTRA_BODY | Дополнительные параметры провайдера; JSON-объект, по умолчанию `{}` |
| TOOL_FILES_ROOT | Каталог доступных файлов; по умолчанию `examples/tool_data` |

Для DeepSeek можно отключить thinking для структурированной финализации:

```dotenv
LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}
```

Примеры входят в Docker-образ. Для своих файлов при `make run` задайте локальный
TOOL_FILES_ROOT. В Docker подключите нужный каталог через Compose override:

```yaml
services:
  api:
    environment:
      TOOL_FILES_ROOT: /tool-files
    volumes:
      - ./my-etl-data:/tool-files:ro
```

Запуск: `docker compose -f compose.yaml -f compose.tools.yaml up -d --build`.
При тестировании используйте стандартный каталог с учебными данными.

## Примеры

```sh
uv run --locked python examples/client.py 'Вычисли (120-90)/120*100 инструментом calculator'
uv run --locked python examples/client.py 'Прочитай etl.log и объясни ошибку загрузки'
uv run --locked python examples/client.py 'Найди invalid date в etl.log через text_search'
uv run --locked python examples/client.py 'Прочитай etl.log, затем calculator вычисли процент отклонённых строк'
```

Содержимое файлов трактуется как данные. Агент должен опираться на observations,
не придумывать проверки и не утверждать, что изменил систему. Итог включает ответ
и следующий шаг диагностики. Протокол обмена доступен в `result.messages`,
а вызовы и результаты — также в серверных логах.

## Ручная проверка

В `.env` настройте `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.
Для учебных примеров используйте `TOOL_FILES_ROOT=examples/tool_data`.
Запустите Docker-стек; локальный `make run` одновременно работать не должен:

```sh
make up
curl http://127.0.0.1:8123/ready
```

Ожидается `{"status":"ready"}`. В другом терминале запустите задачи по очереди:

```sh
uv run --locked python examples/client.py 'Используй calculator для вычисления (120-90)/120*100'
uv run --locked python examples/client.py 'Прочитай etl.log через read_local_file. Сколько строк отклонено?'
uv run --locked python examples/client.py 'Найди invalid date в etl.log через text_search'
```

| Задача | Ожидаемый observation |
|---|---|
| calculator | `result: 25.0` |
| read_local_file | Текст лога с `rejected_rows=30` |
| text_search | Строка 3 с датой `2026-99-21` |

Клиент печатает `task_id` и JSON результата после успешного завершения.
В `messages` найдите сообщение `type=ai` с нужным `tool_calls`, затем
`type=tool` с `status=success`. Его `tool_call_id` должен совпадать с `id`
вызова. Поле `content` содержит observation в виде JSON-строки.
`final_answer` должен учитывать результат, а `next_action` — содержать
действие и обоснование.

Проверьте передачу observation следующему вызову LLM:

```sh
uv run --locked python examples/client.py 'Сначала read_local_file прочитай etl.log. Только после чтения calculator вычисли процент отклонённых строк из expected_rows и rejected_rows.'
```

В `messages` observation чтения должен предшествовать вызову калькулятора.
Ожидаемый итог — 25%.

Логи вызовов и результатов доступны в отдельном терминале:

```sh
make logs
```

Ищите `event=tool.started` и `event=tool.completed`: имя инструмента,
`tool_call_id`, аргументы, статус, длительность и результат.

Чтобы посмотреть SSE, подставьте UUID из вывода клиента вместо `<task_id>`:

```sh
curl -N 'http://127.0.0.1:8123/tasks/<task_id>/events'
```

Ожидаются progress-события `input`, `planning`, `tools`, `finalization`
и итоговый статус `success`. Сохранённые события доступны и после завершения
задачи. Каждое имя узла выводится один раз, даже если planning/tools повторялись.
