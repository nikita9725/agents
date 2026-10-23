# Полные реальные traces

Запуски выполнены 2026-10-23 через публичный FastAPI/SSE API отдельного
стека `etl-agent-test` (порт 18123), с реальной моделью `deepseek-flash`.
LangSmith отключён по выбору пользователя. Файлы содержат локальные traces,
а не выгрузку из облачного LangSmith.

[Пять demo и команды запуска](../../docs/demo.md).

| Demo | task_id | Шаги acting | Причина остановки |
|---|---|---|---|
| 1 | `c57ad588-f721-4189-bb91-68e458e993a5` | 1 | `criteria_met` |
| 2 | `ef78d1ac-a1b7-4be5-acbe-3514dac94871` | 1 | `criteria_met` |
| 3 | `1a16318f-1105-4e33-a1cf-6b68127be956` | 1 | `criteria_met` |
| 4 | `dc1f80b3-8fed-4a3d-9a0c-253db8ffd646` | 2 | `criteria_met` |
| 5 | `9fc83ce6-75c7-4bbe-a706-df09abd9a33d` | 2 | `actor_finished` |

Все пять запусков завершились со status=success. Для demo 5 это технический
статус workflow; недоступный файл остаётся ограничением результата.

[Машиночитаемые итоги пяти запусков](demo-results.json).

## Demo 4

[Полный JSON задачи](demo-4.json) · [Полный SSE](demo-4.sse)

JSON содержит исходную цель, план, все messages/tool calls, results, observations,
notes, criteria/evidence, history/state_delta и финальный ответ. SSE содержит
последовательные progress и терминальный snapshot той же задачи.

Порядок записей history:

```text
input → planning → acting → tools → observation → acting → tools → observation → observation → finalization
```

| Результат | Tool | Status | Аргументы |
|---|---|---|---|
| `result_1` | `read_local_file` | `success` | `{"path": "etl.log"}` |
| `result_2` | `text_search` | `success` | `{"path": "etl.log", "query": "invalid date"}` |
| `result_3` | `calculator` | `success` | `{"expression": "30 / 120 * 100"}` |

**Финальный ответ этого запуска:**

> Итог диагностики ETL-загрузки:
>
> 1) Прочитан etl.log: expected_rows = 120, loaded_rows = 90, rejected_rows = 30.
> 2) text_search по 'invalid date' в etl.log дал 1 совпадение — строка 3 файла:
>    «2026-10-21 10:00:02 ERROR invalid date at row 2: 2026-99-21»
>    Ошибочная дата: 2026-99-21 (некорректный месяц 99), в строке 2 данных.
> 3) Расчёт calculator: 30 / 120 * 100 = 25.0.
>    Процент отклонённых строк от ожидаемых: 25%.
>
> Итого: отклонено 25% строк (30 из 120, загружено 90); причина — невалидная дата 2026-99-21 в строке 2 входных данных.
>
> Важно: указанные проверки — чтение лога, поиск и расчёт; изменения в системе/файлах не вносились.
> Критерии диагностики выполнены по сохранённым наблюдениям.

**Критерии:**

- etl.log прочитан через read_local_file, содержимое доступно для анализа — satisfied=true, evidence=["result_1"].
- Через text_search найдены строки с текстом 'invalid date', зафиксирована ошибочная дата — satisfied=true, evidence=["result_2"].
- Из лога получены значения rejected_rows и expected_rows — satisfied=true, evidence=["result_1"].
- calculator вычислен процент rejected_rows от expected_rows, результат указан — satisfied=true, evidence=["result_3"].
- В ответе указаны вычисленный процент и ошибочная дата — satisfied=true, evidence=["result_3", "result_2"].

## Demo 5

[Полный JSON задачи](demo-5.json) · [Полный SSE](demo-5.sse)

JSON содержит исходную цель, план, все messages/tool calls, results, observations,
notes, criteria/evidence, history/state_delta и финальный ответ. SSE содержит
последовательные progress и терминальный snapshot той же задачи.

Порядок записей history:

```text
input → planning → acting → tools → observation → observation → acting → finalization
```

| Результат | Tool | Status | Аргументы |
|---|---|---|---|
| `result_1` | `calculator` | `success` | `{"expression": "2+2"}` |
| `result_2` | `read_local_file` | `error` | `{"path": "missing.log"}` |

**Финальный ответ этого запуска:**

> Итог по двум запрошенным проверкам:
>
> 1) calculator: 2 + 2 = 4.0 (status=success).
> 2) read_local_file для пути missing.log: ошибка — {"error":"File not found or not a regular file"} (status=error).
>
> Ограничение (фиксирую явно): содержимое missing.log получить не удалось — файл не найден либо не является обычным файлом. Содержимое этого файла я не читал и не воспроизвожу; любые утверждения о его содержимом были бы домыслом. Также ничего в системе не изменялось — выполнялись только чтение и вычисление.
> Не подтверждено: Выполнена попытка чтения файла missing.log через read_local_file; Доступность missing.log зафиксирована явно; при недоступности указано ограничение без выдумывания содержимого

**Критерии:**

- Получен результат арифметического вычисления 2+2 через calculator — satisfied=true, evidence=["result_1"].
- Выполнена попытка чтения файла missing.log через read_local_file — satisfied=false, evidence=[].
- Доступность missing.log зафиксирована явно; при недоступности указано ограничение без выдумывания содержимого — satisfied=false, evidence=[].

## Проверка и повторение

В сохранённых запусках проверены terminal success, наличие плана и final answer,
требуемые tools и факты: 25%, 30 строк, неверная дата, расчёт 4 и ошибка missing.log.
Терминальный SSE snapshot сверён с GET /tasks/{task_id}. Для полного чтения
раскройте JSON/SSE: приведённый выше разбор не заменяет исходные артефакты.

Чтобы повторить, запустите команды [demo](../../docs/demo.md); получатся новые
идентификаторы, план и формулировки. Чтобы сохранить новые trace, используйте
[команды экспорта](../../docs/observability.md#локальная-приёмка-дня-7).

Артефакты используют только учебные данные. Доступность старых task_id через API
зависит от сохранности тестового PostgreSQL volume; сами JSON/SSE автономны.

## Первичная проверка и найденные ограничения

[Первичные результаты](initial-demo-results.json) сохранены отдельно, до последнего
исправления evidence. В первом demo 1 модель добавила неверное числовое ожидание
в критерий, однако финальный ответ сохранил правильные 25% из calculator и явно
отметил неподтверждённый критерий. Первое demo 5 завершилось ошибкой evidence даже
после одной LLM-корректировки. Это привело к детерминированному отклонению
ссылок на неуспешные tools с сохранением критериев невыполненными.

Таблица и demo-4/demo-5 выше относятся к повторной проверке финальной версии.
Ошибки первичного прогона не скрыты; технический success и доказанность всех
утверждений LLM следует оценивать раздельно. Например, строка invalid date
подтверждает ошибку даты, но сама по себе не доказывает, что она объясняет
отклонение всех 30 строк.
