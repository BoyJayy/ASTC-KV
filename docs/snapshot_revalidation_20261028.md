# Полное повторение snapshot-пилота на Mac

Дата: 2026-10-28. Статус: **Выполнено и проверено** в пределах CPU snapshot. Все 936 строк получены новым запуском кодирования и attention, а не пересчётом сохранённых средних. Полная модель, новый holdout и скорость GPU этим опытом не проверялись.

Результат: **все 117 исходных screening decisions сохранились**, все 936 memory dictionaries совпали точно. Максимальное абсолютное расхождение row-relative-L2 - `2.68865914e-8`; агрегированного relative-L2 - `7.49415840e-9`. Численный вывод прежнего snapshot-пилота воспроизводится: scalar6/token V-only проходит предварительный screening, U4 V-only имеет меньшую ошибку, asymmetric U4 K+V сохраняет значительно меньший полный KV-бюджет.

## Исполнение и происхождение

Использованы исходные `scripts/run_pilot.py`, `scripts/summarize_pilot.py`, `configs/pilot_v1.json`, `astc_kv/codec.py`, `bridge.cpp` и четыре authentic captures. Их локальные SHA256 вместе с четырьмя исходными JSONL проверены против manifest неизменённого архива. SHA256 архива: `13ddaaa57a0b2c81457155aeece426e1977c3bc3a805bca0f8f0988d77e60b95`. Все 13 проверенных файлов совпали. Доказательство: [archive_identity.json](../results/revalidation_20261028/snapshot/archive_identity.json).

Python: `.venv-research/bin/python`, NumPy 2.3.5. Использован реально собранный Mac `astc_kv/libakv_astc.dylib`, SHA256 `ffbf5f0324cc639b5dd315dbe256ca952d6b66994886a14e6b35615eed32166b`. Кодировка - настройки исходного pilot, включая astcenc quality 98. Полные Python/platform сведения, source/binary/input hashes и команды находятся в [run_manifest.json](../results/revalidation_20261028/snapshot/run_manifest.json). Все хеши оставались неизменными от начала до конца серии.

Четыре дочерних процесса запускались **последовательно**, с `OPENBLAS_NUM_THREADS=1` и `VECLIB_MAXIMUM_THREADS=1`. Перед созданием нового каталога проверено его отсутствие. Старые raw results не перезаписывались. Для воспроизведения следует выбрать другой новый каталог: сам исходный `run_pilot.py` открывает `--out` на перезапись.

| Capture | Exit code | Новых строк | Wall seconds |
|---|---:|---:|---:|
| systems | 0 | 234 | 10.559769 |
| story | 0 | 234 | 10.633059 |
| science | 0 | 234 | 10.662506 |
| records | 0 | 234 | 10.772051 |

Полная последовательность с агрегацией заняла 42.720076 s. Эти длительности описывают выполнение CPU-проверки и не служат сравнением производительности ASTC/low-bit runtime. CPU-кодирование завершено до передачи машины для дальнейших аппаратных timing.

Фактически исполнены четыре команды следующего вида, по порядку systems, story, science, records:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/run_pilot.py --capture results/captures/systems.npz --config configs/pilot_v1.json --out results/revalidation_20261028/snapshot/pilot_systems.jsonl
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/run_pilot.py --capture results/captures/story.npz --config configs/pilot_v1.json --out results/revalidation_20261028/snapshot/pilot_story.jsonl
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/run_pilot.py --capture results/captures/science.npz --config configs/pilot_v1.json --out results/revalidation_20261028/snapshot/pilot_science.jsonl
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/run_pilot.py --capture results/captures/records.npz --config configs/pilot_v1.json --out results/revalidation_20261028/snapshot/pilot_records.jsonl
.venv-research/bin/python scripts/summarize_pilot.py results/revalidation_20261028/snapshot/pilot_systems.jsonl results/revalidation_20261028/snapshot/pilot_story.jsonl results/revalidation_20261028/snapshot/pilot_science.jsonl results/revalidation_20261028/snapshot/pilot_records.jsonl --out results/revalidation_20261028/snapshot/pilot_summary.json
```

Окружение с ограничениями числа потоков наследовалось также summarizer. Стандартный вывод и ошибки каждого запуска сохранены в соответствующих `.log`, дополнительно исходный скрипт сохранил `.provenance.json`. Полные абсолютные аргументы процесса зафиксированы в run manifest.

## Независимое сравнение

Новый [compare_snapshot_reproduction.py](../scripts/compare_snapshot_reproduction.py) использует только Python standard library; не импортирует кодек, attention или исходный summarizer. Он сопоставляет строки по `(capture, layer, variant, mode)`, сравнивает все числовые поля metrics/tensor errors/per-head errors/SSE, отдельно проверяет memory и структуру, повторно рассчитывает 117 агрегатов через суммы SSE и исходные gates.

```sh
.venv-research/bin/python scripts/compare_snapshot_reproduction.py --original-dir results --reproduced-dir results --out results/revalidation_20261028/snapshot/comparator_identity.json
.venv-research/bin/python scripts/compare_snapshot_reproduction.py --original-dir results --reproduced-dir results/revalidation_20261028/snapshot --out results/revalidation_20261028/snapshot/comparison.json
```

Оба запуска завершились с exit code 0. Identity control дал ровно нулевые расхождения. Для Mac reproduction сравнены **97 344 числовых значения**, у 77 657 различаются последние разряды. Большое число отличающихся floating-point значений не означает большого эффекта: величины различий приведены ниже. Никакой допуск для объявления побитового совпадения постфактум не подбирался.

| Проверка | Результат |
|---|---|
| 936 уникальных идентификаторов строк и сопоставимость структуры | PASS |
| Конечность всех сравниваемых числовых метрик | PASS |
| Полные memory dictionaries | 936/936 точно совпадают |
| Screening decisions | 117/117 совпадают |
| Новый summarizer против независимой агрегации | max relative-L2 difference 0; gates совпадают |
| Побитовое совпадение ASTC payload | Не проверялось: исходный pilot JSONL не содержит payload |
| Подтверждение качества модели/PPL новым запуском | Не входит в этот опыт |

Основные сохранённые выводы на историческом pilot holdout:

| Вариант | Исходный relative-L2 | Новый relative-L2 | Бит / весь KV | Gate |
|---|---:|---:|---:|---|
| scalar6_token, V_only | 0.07425469323115132 | 0.07425469259359645 | 11.5625 | pass -> pass |
| u4_token, V_only | 0.04039357218176319 | 0.04039357244853170 | 11.6875 | pass -> pass |
| u4_asymmetric, KV | 0.06645038004152973 | 0.06645038052383499 | 7.3125 | pass -> pass |

Исторические science/records называются holdout только относительно первоначального выбора. Их повторное использование в этой серии не создаёт новый независимый тест.

## Максимальные различия и ограничения объяснения

- Row relative-L2: максимум `2.68865914e-8`, records/layer23/u2_token/KV: `1.5369501591786636 -> 1.536950186065255`.
- Aggregate relative-L2: максимум `7.49415840e-9`, development/u2_token/K_only: `0.8679997175021998 -> 0.8679997249963582`.
- Error SSE: максимальная абсолютная разность `5.41994677e-5`, story/layer23/u2_token/K_only: `4312.157020797206 -> 4312.157074996674`.
- Error SSE: максимальная симметричная относительная разность `1.05975121e-4`, records/layer0/fp16/KV, при очень малой самой SSE: `5.72683413648e-8 -> 5.72622723454e-8`, абсолютная разность `6.06901943e-12`.
- Reference SSE: максимальная симметричная относительная разность `4.40829256e-8`.
- Row mean KL: максимальная абсолютная разность `1.29532169e-8`, records/layer23/scalar6_hdr_block/KV. Относительные расхождения KL возле нуля заметно больше и не должны интерпретироваться без абсолютной шкалы.
- Tensor relative-L2 K и V: максимальные абсолютные разности обе равны `1.33226763e-15`.

Различия присутствуют также у FP16 и affine controls. Это согласуется с различиями floating-point reductions/линейной алгебры между прежним Linux и текущим Mac. Причина не изолирована отдельным экспериментом, поэтому приписывать остаточные отличия именно SIMD-выбору ASTC modes нельзя. Совпадение tensor-error метрик до примерно 1e-15 также не доказывает идентичности каждого decoded scalar или bitstream.

Полная таблица максимумов для каждого числового поля, 20 худших строк по абсолютной и симметричной относительной разнице, статистика каждой representation и все 117 сравнений находятся в [comparison.json](../results/revalidation_20261028/snapshot/comparison.json).

## Решение

Сохранить численный snapshot-вывод v0.1: он выдержал новое кодирование на другом CPU/backend окружении. Это усиливает доверие к ограниченному эксперименту, но не меняет его узкую область: те же четыре текста, шесть слоёв, T384, статический префикс 288, без обратной связи ошибок через всю модель. Следующие решения о runtime должны опираться на отдельный аппаратный контроль, сильный low-bit baseline и новый модельный эксперимент, а не на сам факт успешного повторения.
