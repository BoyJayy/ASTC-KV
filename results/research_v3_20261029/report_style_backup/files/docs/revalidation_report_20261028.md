# ASTC-KV: независимая перепроверка и продолжение

Дата: 28 октября 2026. Архив v0.1 сохранён неизменным. Новые исходники и данные находятся в этом проекте; результаты этой серии — в `results/revalidation_20261028/`.

## Вывод для проекта

Предыдущий CPU-пилот воспроизводится. Впервые выполнен прямой ASTC texture read внутри attention на Apple M5 Pro. Установлены и исправлены ошибки переносимости сборки и ошибка общего входа в MLX benchmark. Проверены две исследовательские идеи: средняя поправка V не помогает лучшему ASTC-варианту; U4 K + ASTC V проходит прежний пилотный PPL screen, но экономит только 1,72% логических байтов относительно асимметричного U4.

Прорыв, преимущество полной генерации и мировая новизна не установлены. Текущий результат полезен как воспроизводимая проверка осуществимости и ограничений. Быстрый GPU encoder пока не оправдан: сначала нужен более выгодный предсжатый attention-путь против сильного low-bit baseline.

## Что перепроверено независимо

- SHA256 ZIP совпал с мастер-промптом; проверены 113 файлов его manifest и CRC архива. Локальные результаты, использованные для пересчёта, связаны с архивом проверкой байтов. [Аудит](../results/revalidation_20261028/audit_v2.json).
- Заново пересчитаны 117 агрегатов из 936 snapshot rows и модельные NLL из 3040 loss records. Ошибки прежней агрегации не обнаружены; максимальная разность mean NLL — `8,88e-16`.
- Заново выполнены все 936 snapshot rows с настоящим ASTC-кодированием на Mac. Все 117 решений screening и все 936 таблиц памяти совпали; максимальное изменение aggregate relative-L2 — `7,49e-9`. [Отдельный отчёт](snapshot_revalidation_20261028.md).
- Заново выполнены 2660 основных и 380 дополнительных модельных предсказаний на всех 30 слоях. Четыре greedy-последовательности совпали с архивом полностью. Модельные агрегаты не бит-идентичны: наибольшая разность mean NLL — `0,001624` у неудачного ASTC page K+V; наибольшая разность отдельной loss — `0,119568`. Все семь первоначальных aggregate PPL gates сохранились. [Сравнение](../results/revalidation_20261028/model_comparison.json).
- Mac NumPy forward проверен против сохранённых Linux Q/K/V/hidden всех четырёх текстов: все исходные допуски `max_abs<1e-3`, `relative_L2<1e-4` пройдены. Incremental self-check на 32 токенах также прошёл. Это не новая проверка против Hugging Face. [Данные](../results/revalidation_20261028/mac_forward_validation.json).
- 12 unit checks пройдены: прежние 7, включая реальный ASTC roundtrip, и 5 новых проверок причинного разложения ошибки. [Лог](../results/revalidation_20261028/numeric_tests_v2.txt).

Различия полной модели согласуются с различиями арифметики между платформами и чувствительностью дискретного кодирования к малым изменениям входа, но конкретная причина каждого отличия не изолирована. Нельзя называть весь model rerun бит-идентичным.

## Модельные числа на текущем Mac

Одинаковые BF16 веса, расширенные в FP32; четыре прежних текста, по 95 target losses, все 30 слоёв, prefix288 и точный FP16 tail95 у последнего предсказания. Это CPU quality simulation с реальной обратной связью ошибок через модель; кеш хранится восстановленным. Биты и байты ниже логические, а не GPU allocation.

| Политика | PPL относительно FP16 | Изменение PPL | Байты всего KV |
|---|---:|---:|---:|
| FP16 | 1,000000 | 0% | 8 824 320 |
| FP16 K + U4 V | 1,001885 | +0,1885% | 6 439 680 |
| FP16 K + ASTC6 token V | 1,010208 | +1,0208% | 6 370 560 |
| Асимметричный U4 K+V | 1,004144 | +0,4144% | 4 020 480 |
| **Новый фиксированный U4 K + ASTC6 token V** | **1,013564** | **+1,3564%** | **3 951 360** |

Новый гибрид зафиксирован в [конфигурации](../configs/hybrid_u4k_astcv_20261028.json) до его запуска, но выбран после просмотра прежних данных. Он экономит 69 120 байт, или **1,7192%**, относительно asymmetric U4 и ухудшает PPL относительно него на **0,9381%**. Проходит прежний aggregate screen ≤2%; отдельно `systems` даёт +2,0250%, `story` +2,0149%. Нельзя задним числом вводить обязательный per-document gate. [380 новых loss records и итог](../results/revalidation_20261028/hybrid_model_quality.json).

Проверяемое ограничение бюджета: при фиксированных U4 K (898 560 байт) и FP16 tail (2 188 800 байт) даже нулевая стоимость V-префикса дала бы только **23,2092%** экономии относительно U4 K+V в этом T383 опыте. Цель ≥25% здесь недостижима изменением одного V при неизменных K и хвосте. Это арифметическое ограничение данного режима, не новая теорема и не запрет выигрыша на более длинном контексте или по скорости.

## Новый отрицательный опыт: согласованная ошибка V

[Постановка](../configs/revalidation_h2a_20261028.json) записана до новых результатов. При фиксированном причинном P использовано точное `ΔO=P·E`. Проверены analytic sign-null при сохранении каждой нормы строки, 256 знаковых и 256 перестановочных контролей на исходном fixture, разложение `E=B+R` с cross-term, затем все 24 сочетания документ×слой. Дополнительные сведения по 9 Q-головам сохранены. Коррекция использует только завершённый префикс и его reconstruction, без будущих Q.

| V-метод | Исходный relative-L2 | После FP16 page-mean correction | Изменение SSE |
|---|---:|---:|---:|
| ASTC6 token | 7,6744% | 7,8729% | +5,2415% |
| ASTC6 page | 23,7458% | 23,4642% | −2,3575% |
| U4 token | 4,1674% | 4,2646% | +4,7200% |

У лучшего ASTC-варианта коррекция ухудшает совокупную ошибку **на каждом из четырёх документов**. Добавляет 0,1667 bit/V; token6 становится 4,3333 bit/V, а corrected U4 — 4,6667 и служит отдельным контролем вне исходного бюджета 4,5. Критерий продвижения коррекции не пройден. [Результаты с provenance и headwise данными](../results/revalidation_20261028/h2a_all_v2/analysis.json).

При этом положительный cross-token excess есть в 20 из 24 layer cases token6. Значит, гипотезы «есть корреляционная структура ошибки» и «достаточно вычесть среднее страницы» различны: первая согласуется с измерением, вторая не дала полезного действия. Это exploratory результат на просмотренных данных, не доказанная новая общая закономерность.

## Впервые проверенный аппаратный путь

Устройство: Apple M5 Pro, 20 GPU cores, 48 GB; macOS 26.6.2, SDK 26.5. Полный Xcode/офлайн `metal` compiler отсутствуют. Host собран установленными Command Line Tools; MSL 3.0 компилируется через публичный runtime API с отключённым fast math. Никаких системных установок не потребовалось.

Два настоящих fixture: story/layer17, T384, Hq9/Hkv3, D64; scalar6 token V-only и scalar6 page K+V. ASTC blocks загружаются в compressed texture; attention читает их напрямую. Диагностический полный decode нужен только для независимой сверки и не входит в timed attention.

- Attention ASTC vs FP64 reference на тех же GPU-decoded значениях: **4,012e-7** и **4,571e-7** relative-L2.
- GPU и astcenc software decode **не идентичны**. V differences relative-L2: 0,001086 и 0,001014; K page: 0,000668. На нормализованных значениях max_abs ≈0,00048733. Простой FP16 cast software values не объяснил отличие. Полная причина пока не установлена.
- При подстановке hardware-decoded prefix в causal snapshot layer17 V-only relative-L2 меняется 9,9133%→9,9081%, page K+V 31,2159%→31,2160%. Это ограниченная проверка влияния, не GPU model PPL. [Сравнение](../results/revalidation_20261028/metal/decode_comparison.json).
- Физическая ASTC texture для V занимает **49 152 байта** при payload **33 792**. Весь persistent V-only cache + metadata — **201 216** против **185 856** logical bytes. Scratch, staging и другие одновременно живущие контроли считаются отдельно. Production peak не измерен.

После первого readback введён отдельный инженерный decode gate: `abs_delta <= scale/1024 + FP32_roundoff`. Он **калиброван по этим fixture**, не является заранее заданным научным quality criterion или доказанной границей ASTC. FP16-сторона проверяется точно. Отсутствие reference-поля, NaN reference и грубо неверный reference приводят к отказу до публикации timing.

Финальные проверки: два исходных fixture + производные K-only и FP16 прошли; три отрицательных контроля дали ожидаемый exit1 и пустой список raw_samples. Итого **7/7 ожидаемых исходов**. [Протокол с точными командами и хешами](../results/revalidation_20261028/metal/final_controls/validation.json). Все возможные D, footprints и RGBA не проверены.

Диагностические 30-round замеры сохранены отдельно. Малый повторно используемый T384 cache, отсутствие интеграции с моделью и разные значения/ветки контролей не позволяют объявить bandwidth или inference speedup. В первой финальной серии ASTC V-only ≈80,90 мкс GPU/attention, обычная texture ≈81,93, простой FP16 buffer ≈43,10. ASTC путь здесь медленнее буфера; близость двух texture paths сама по себе не локализует причину между адресацией, metadata, вычислениями и texture accesses. Серию могли затронуть параллельные CPU-проверки; отдельный завершающий launcher сохраняет новую серию после них. Отношение этих batched GPU timings к MLX host timings не вычисляется.

Полный обновлённый launcher отдельно завершился exit0 после тяжёлых CPU-опытов.
В его V-only серии medians составили **81,035 / 82,137 / 43,281 мкс GPU** для
ASTC / ordinary texture / FP16 buffer. Это подтверждает наблюдаемый проигрыш
именно текущего texture-пути в этом малом режиме. [Финальная сводка](../results/revalidation_20261028/metal/launcher_final/story_l17_v_token6_summary.json).

## Исправлена ошибка MLX benchmark

Установлены локально MLX 0.32.4 и MLX-LM 0.32.2. В вызываемой функции есть `queries *= scale`, который фактически изменяет переданный объект Q. Минимальный синтетический GPU опыт дал `1.0→0.125`. Прежний harness переиспользовал Q, поэтому warmup и последующие варианты меняли вход. Это ошибка контракта **нашего стенда**, не доказательство ошибки обычного inference MLX-LM, где Q обычно свежий.

Вызов теперь получает свежий `q+0` expression. До timing проверяются неизменность общего Q, одинаковые результаты повторных вызовов, finite и independent FP64 reference на собственных reconstructed KV. Все 5 ветвей прошли; максимальный relative-L2 **0,00055153**, зафиксированный до проверки допуск 0,005. Сохранены первоначальный fail, минимальное воспроизведение и успешные 50 repeats/ветвь. [Успешный baseline](../results/revalidation_20261028/mlx_baseline.json).

FP16 fastSDPA median host latency ≈172,1 мкс; quantized paths ≈199–207 мкс на одном маленьком all-cache fixture. Это host dispatch + completion, не чистый GPU time. Q4 quality в этом конкретном all-KV fixture существенно хуже FP16; вычислительный correctness gate не является quality certificate. Форматы, точность Q, политика K/V и timing boundary отличаются от ASTC: прямое отношение скоростей невалидно.

## Что меняется в научной постановке

[Независимая рецензия](independent_research_20261028.md) повторно открывает первичные источники Apple, LLM.265, KVCodec, KIVI, TurboQuant, Open-TQ и текущие MLX/llama.cpp исходники. Добавлены [KVLinC](https://arxiv.org/html/2510.05373v1), [Quantized Keys Steal Attention](https://arxiv.org/html/2605.26266v1), [eOptShrinkQ](https://arxiv.org/html/2605.02905v1). Коррекция bias, per-token V, low-rank residual и само использование аппаратного codec для KV не являются достаточной новизной.

Узкая ещё открытая возможность: представление и исполнение активного KV, совместно спроектированные под ASTC texture decode, с полезным режимом против сильного low-bit runtime. Точный аналог не найден в выполненном целевом поиске; мировое первенство не доказано.

Следующий различающий этап: сопоставить ASTC, normalized FP16 buffer с той же affine reconstruction и ordinary texture на одинаковых данных/Q и общих timing boundaries. Это отделит стоимость decoder от остальных операций общего texture path. Затем проверить реальный fused Q4 dispatch llama.cpp и достаточно большой working set. Если предсжатый путь не даёт перспективы, GPU encoder не разрабатывать; менять представление через фиксированные внутриблочные перестановки/H1a с отрицательным контролем перестановки целых блоков. Новый алгоритм подтверждать на новых документах и затем второй модели.

## Проверки, команды и невыполненные этапы

Рабочий каталог всех команд — корень `astc-kv`. Результаты в перечисленных путях уже существуют; для повторения нужны новые output paths.

```sh
bash scripts/build_codec.sh
.venv-research/bin/python -m unittest discover -s tests -v
.venv-research/bin/python scripts/audit_handoff.py --archive /home/researcher/Downloads/ASTC-KV_v0.1.zip --out results/revalidation_20261028/audit_v2.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/evaluate_model.py --out-prefix results/revalidation_20261028/model_quality
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/evaluate_model_secondary.py --primary results/revalidation_20261028/model_quality.json --out-prefix results/revalidation_20261028/model_quality_secondary
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/analyze_v_error.py --all-captures --out-dir results/revalidation_20261028/h2a_all
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/analyze_v_error.py --all-captures --reconstructions-dir results/revalidation_20261028/h2a_all --out-dir results/revalidation_20261028/h2a_all_v2
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python scripts/evaluate_hybrid.py --primary results/revalidation_20261028/model_quality.json --secondary results/revalidation_20261028/model_quality_secondary.json --out-prefix results/revalidation_20261028/hybrid_model_quality
bash metal/build.sh
.venv-research/bin/python scripts/verify_metal_controls.py --out-dir results/revalidation_20261028/metal/final_controls
.venv-research/bin/python scripts/compare_hardware_decode.py --out results/revalidation_20261028/metal/decode_comparison.json
.venv-research/bin/python scripts/probe_mlx.py --capture results/captures/story.npz --layer 17 --warmup 10 --repeats 50 --out results/revalidation_20261028/mlx_baseline.json
PYTHON=.venv-research/bin/python OUT_DIR=results/revalidation_20261028/metal/launcher_final bash scripts/run_mac_probe.sh
git diff --check
```

PASS: CPU build после исправления `ASTCENC_UNIVERSAL_BUILD=OFF`; реальные численные тесты и опыты; MSL runtime compilation; аппаратные storage controls; ожидаемые отрицательные проверки; MLX correctness после исправления Q. Snapshot-команды и источники результатов отдельно приведены в связанном отчёте.

FAIL, сохранено: исходный Mac codec configure (несовместимый universal/native); исходный offline Metal build (нет compiler); sandbox GPU access; initial MLX shared-Q correctness; первый syntax check нового диагностического скрипта (исправлен до опытов); решение продвигать page-mean correction.

SKIPPED / ещё не реализовано: fresh Hugging Face forward validation, новый независимый корпус, второй checkpoint, длинный causal rollout с повторными событиями сжатия, новые footprints/D/RGBA на GPU, llama.cpp fused Q4 hardware run, GPU encoder, dynamic active-cache integration, end-to-end latency/production peak и статистическое подтверждение выигрыша. Это явные следующие этапы; весь исследовательский проект завершённым не объявляется.
