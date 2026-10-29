# Подтверждающая проверка фиксированного transpose

Дата: 2026-10-28. Статус: полный прогон завершён, заранее заданный критерий не пройден. Конфигурация и четыре новых текста с токенами были зафиксированы до inference. Направление transpose выбрано на предыдущем exploratory H1a; здесь ориентации не перебирались.

## Подтверждающий результат

**Transpose не подтвердил улучшение PPL:** отношение aggregate PPL transpose/identity составило **1.0011360376642466** при заранее заданном максимуме 1.0. Ухудшение — 0.1136%; разность mean NLL — около +0.00113539 nat/target. Это FAIL фиксированного критерия, а не доказательство статистически устойчивого вреда столь малого изменения.

Все 1520 target losses (380 на каждый из четырёх методов) получены, все контроли прошли. Wall time полного CPU-опыта — 37.5163 s; ранее существовавшие файлы не изменялись. Новая серия включает все 30 слоёв с реальным error feedback.

| Метод | Mean NLL | PPL | PPL/FP16 | Логические bytes при последнем prediction |
|---|---:|---:|---:|---:|
| FP16 | 3.432309007 | 30.948019508 | 1.000000000 | 8 824 320 |
| ASTC identity V-only | 3.445500726 | 31.358981789 | 1.013279114 | 6 370 560 |
| ASTC transpose V-only | 3.446636119 | 31.394606773 | 1.014430237 | 6 370 560 |
| Asymmetric U4 K/V | 3.435178430 | 31.036949981 | 1.002873543 | 4 020 480 |

| Новый документ | PPL transpose/identity |
|---|---:|
| ceramic_workshop | 1.002354893 |
| coastal_letter | 1.002415485 |
| bread_notes | 1.004664137 |
| choir_rehearsal | 0.995135451 |

Transpose хуже identity на трёх документах и лучше на одном; per-document пороги не вводились. Оба ASTC метода расходуют одинаковые cache bytes: FP16 K-prefix 3 317 760, ASTC V-prefix 864 000, FP16 tail 2 188 800. Это на 27.8068% меньше FP16 K/V, но на 58.4527% больше asymmetric U4 K/V. V-only ASTC здесь сохраняет K в FP16 специально для чистого сравнения ориентаций; результат нельзя подменять памятью прежнего U4-K/ASTC-V combo.

Прежний exploratory выигрыш snapshot attention на старых captures не перенёсся в меньший causal teacher-forced PPL на этой новой серии. Одновременно изменились данные и метрика, поэтому опыт не разделяет domain shift и несовершенство snapshot error как surrogate. Оснований заменять identity на transpose в основном codec после этого результата нет. Пороги, тексты и варианты не менялись; нового поиска по этим документам не проводилось.

## Заранее фиксированный протокол

Модель — тот же SmolLM2-135M, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, исходные BF16 weights точно расширяются до FP32. Используются все 30 слоёв. Каждый новый текст токенизируется штатным локальным tokenizer без BOS/chat template; берутся первые 384 токена. Только префикс позиций 0..287 проходит prefill и однократное кодирование. Далее потребляются позиции 288..382 с реальным влиянием предыдущих ошибок на последующие слои и шаги; новые K/V округляются в FP16. Оцениваются 95 targets 289..383 на текст, всего 380 на метод. Target 288 исключён, потому что был предсказан до изменения cache.

Четыре метода зафиксированы до результата:

- FP16 K/V.
- ASTC6 token-normalized V с исходной ориентацией; K остаётся FP16.
- Та же ASTC V с единственным фиксированным transpose каждого нормализованного блока 6×6; K остаётся FP16.
- Асимметричный affine U4: K/channel/group96, V/token/group64.

ASTC LDR scalar RGB, quality98; ranges вычисляются до transform, FP16 metadata и edge-padding D64→66 совпадают. После decode применяется обратный transpose, затем исходные token ranges. Для этой глобальной перестановки нет дополнительных cache bytes. U4 K/V — отдельный comparator с другим размером cache, а не утверждение о сопоставимой памяти с V-only ASTC.

**Единственный основной критерий:** `exp(meanNLL_transpose − meanNLL_identity) ≤ 1.0` по всем 380 совпадающим targets, при полном конечном результате. Порог, методы и документы после результата не меняются. Per-document значения, PPL относительно FP16/U4 и логическая память — вторичные описательные результаты без дополнительных порогов принятия.

Новые тексты написаны специально для этой проверки, их полные варианты и выбранные token IDs сохранены в `fixtures/geometry_holdout_v2/`. Это небольшая подтверждающая серия после выбора гипотезы, не репрезентативный benchmark и не доказательство устойчивого качества на неизвестных задачах. Сюжеты и token sequences отличаются от прежних systems/story/science/records; это проверяется до запуска. Все четыре текста остаются authored English prose одного маленького модельного checkpoint.

| Документ | Токенов в полном тексте | Выбрано |
|---|---:|---:|
| ceramic_workshop | 428 | 384 |
| coastal_letter | 449 | 384 |
| bread_notes | 441 | 384 |
| choir_rehearsal | 448 | 384 |

Полные SHA256 текстов, файлов token IDs и канонических little-endian uint32 IDs находятся в `configs/geometry_holdout_v2.json`. Config SHA256 до inference: `722a3e90541d99f30ea6d92cb8e9a159f935f0531ec56c4a00810e04cbb99fd4`. Код, codec, bridge, dylib, tokenizer и model config также закреплены hashes. Weight SHA берётся из предыдущего проверенного локального checkpoint и сверяется с фактическим файлом до и после запуска.

## Контроли и ресурсы

Подготовительная проверка PASS: transpose сопоставлен с независимым циклом явного отображения координат; проверены инволюция, сохранение block multiset, обратимость с разными token ranges и padding, исключение target до изменения cache, округление только нового FP16 tail и неизменность исходного prefix. Она не запускала encoder или модель. После разрешённого старта дополнительный маленький synthetic codec control должен точно воспроизвести payload/decode/metadata существующего identity encoder. Для каждой реальной пары identity/transpose проверяются равенство ranges и byte counts. Подготовленные cache arrays read-only; NumpyLlama.step создаёт новые caches.

Оценка времени до запуска: в предыдущем Mac model-quality прогоне 95 teacher-forced targets занимали median 0.7146 s на метод (28 старых измерений, диапазон 0.7006–0.7630 s). Для 16 новых document/method комбинаций это около 11.4 s только continuation. Старое V page encoding занимало около 3.3 s на 30 слоёв одного документа; два ASTC варианта на четырёх документах дают ориентир около 26 s плюс prefill/IO. Ожидание полного нового прогона — порядка 40–90 s, а не serving performance forecast. Фиксированный wall budget 300 s проверяется между методами; превышение оставляет partial raw results без основного вердикта. Один CPU-процесс, BLAS/VECLIB threads=1; GPU не используется, скачиваний нет.

Точные выполненные команды подготовки:

```sh
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv-research/bin/python scripts/evaluate_geometry_holdout.py --mode freeze
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv-research/bin/python scripts/evaluate_geometry_holdout.py --mode check
```

Первый `freeze` уже выполнен и повторно перезапись не допускается. Полный прогон выполнен после CPU-окна:

```sh
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv-research/bin/python scripts/evaluate_geometry_holdout.py --mode run
```

Все новые результаты записаны только в `results/geometry_holdout_v2/`: frozen config, provenance, 1520 raw losses, summary, per-layer identity/transpose payloads и общие FP16 ranges. Дополнительный actual synthetic codec control точно совпал с прежним encoder по payload/decode/metadata. Source/config/input/weight hashes проверены до и после inference.

Независимая stdlib-проверка заново вычислила mean NLL и все PPL ratios из raw losses через `math.fsum`/`math.exp`: результаты совпали точно. Проверены matching positions/target IDs, layer coverage, payload hashes, исходные document/token/source/config hashes и арифметика памяти. Результат сохранён в `independent_validation.json`; воспроизводимая проверка без inference:

```sh
.venv-research/bin/python results/geometry_holdout_v2/verify_saved.py
```

Статус проверки артефактов: PASS. Статус основного критерия гипотезы: FAIL. Исходные captures, fixtures, core и прежние результаты не менялись. Измеряется качество CPU реконструированного cache и логический byte count; runtime allocation и GPU latency этим опытом не проверяются.
