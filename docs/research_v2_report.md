# ASTC-KV: продолжение исследования v2

Дата: 2026-10-28. Устройство: Apple M5 Pro, 20 GPU cores, 48 GB, macOS 26.6.2. Исходный ZIP, результаты v0.1 и `results/revalidation_20261028/` сохранены. Этот отчёт описывает новые выполненные эксперименты после независимой перепроверки.

## Основной результат

Получено существенно более быстрое **исследовательское** ASTC attention-ядро, проверенное на настоящем Metal. Найдена и проверена другая физическая упаковка тех же ASTC-блоков: на T8192 она уменьшила V texture allocation на 53.125% с побитно тем же hardware decode, но добавила около 14.5% GPU latency к прежнему ASTC layout. Это проверенный обмен памяти на время. Перестановка значений внутри ASTC-блока дала небольшой выигрыш snapshot attention на старых данных, но **не прошла заранее заданную проверку PPL на новых текстах**. Реальный fused Metal baseline llama.cpp собран и выполнен; его первоначальное расхождение с эталоном объяснено через контракт округления Q/K/V.

Прорыв, преимущество полного decode и научная новизна пока не установлены. Ускорение собственного исходного прототипа нельзя представлять как победу над существующим runtime. Числа ниже относятся к разным явно указанным границам измерения.

## 1. Что ограничивало исходное ядро

Первый замороженный sweep менял только размер context split в прежнем ядре. T384 - настоящий исходный fixture; T2048/T8192 повторяют его блоки и являются **контролями трафика и формы**, а не модельными опытами на длинном контексте.

При T384 переход tile256->tile16 снизил GPU time ASTC с 81.351 до 11.989 us. Обычная texture и FP16 buffer также ускорились; лучший FP16 buffer дал 8.897 us. Значит значительная часть старой задержки относилась к последовательному циклу/разбиению работы, а не доказывала фундаментальную цену ASTC.

Следующий контроль v2 сравнивал одинаковый алгоритм и близкие/одинаковые представленные значения: ASTC texture; обычную scalar texture; восстановленный FP16 buffer; нормализованный FP16 buffer с тем же affine; обычную RGBA16 texture с теми же нормализованными значениями. Это **не** опыт с lossy RGBA ASTC.

При T384/tile16 медианы трёх сеансов: ASTC 11.769, normalized buffer 11.810, ordinary RGBA 11.642, restored buffer 8.956 us. Перенос значений из texture в normalized buffer сам по себе не устранил задержку. Это поддерживает объяснение о существенной цене восстановления/организации вычисления; аппаратный ALU/texture bottleneck непосредственно счётчиками не установлен.

Артефакты: `configs/kernel_split_v2.json`, `configs/kernel_attribution_v2.json`; `results/kernel_v2/split_v1_page/`, `attribution/`; [дизайн контролей](kernel_attribution_design_v2.md).

## 2. Алгебра и параллельное исполнение

v3 использует для scalar per-token V тождество

`sum_t p_t (s_t u_t + o_t) = sum_t (p_t s_t) u_t + sum_t p_t o_t`.

Offset накапливается отдельно до завершения split. Для page/channel normalization и FP16 остаётся прежняя ветка. Это дистрибутивность, не новый математический результат. FP32 порядок округления изменился, поэтому каждая ветка проверяется на CPU FP64 reference представленных значений.

v4 обрабатывает восемь токенов за SIMD wave: четыре lanes на токен, 16 каналов на lane, quad reduction для QK и cross-quad reduction для V. Это специализация D64; другие D явно отклоняются. Split/merge softmax сохраняется. Перенос affine и новый schedule проверены раздельными сериями.

| T | Лучший ASTC v1 после split tuning, us | Лучший ASTC v4 + factored V, us | Лучший собственный FP16 v4, us |
|---:|---:|---:|---:|
| 384, реальный fixture | 11.989 | 7.907 | 5.855 |
| 2048, повтор блоков | 27.769 | 15.652 | 10.917 |
| 8192, повтор блоков | 73.931 | 38.370 | 25.393 |

Это медианы медиан двух сеансов, выбранные минимумы exploratory tile sweep. Они не являются заранее отложенным подтверждением оптимального tile. Полные варианты и выбросы сохранены. От исходного tile256/T384 ускорение около 10.3x; от уже настроенного v1 - около 1.52x. **В том же v4 собственный FP16 всё ещё быстрее ASTC.**

Все четыре серии: 90 process runs, 13 860 сырых timed batches, 10 iterations/batch. Максимальная relative L2 против собственного reference - 1.77e-6 при gate 1e-4. Отдельные v4 контроли: 8/8 PASS, включая T391/tile13 с неполной wave, V-only, K-only, K+V/page-channel, FP16 и три отказа до timing на повреждённых references. Независимый read-only review lane reductions не нашёл ошибки. Три дополнительные алгебраические проверки покрывают zero scale, signed offsets, extreme scores и partial splits; это не исчерпывающая проверка всех FP32 cancellation cases.

Артефакты: `metal/attention_v3.metal`, `attention_v4.metal`, соответствующие hosts/build scripts; `results/kernel_v2/factored/`, `wave/`, `v3_controls/`, `v4_controls/`, `derived_kernel_summary.json`.

## 3. Геометрия ASTC: наблюдение и неудачное подтверждение

На старых четырёх captures/шести слоях проверены все восемь D4 симметрий 6x6 блока, fixed random36 и whole-block permutation->inverse. Ranges, состав блока и payload сохраняли размеры. Полный bit budget одинаков для D4; диагностические random36 и whole-block permutation дополнительно учитывают соответственно 36 B и 1056 B mapping metadata. 240 rows; все строгие контроли PASS. Whole-block inverse дал побайтно прежние payload/decode во всех 24 парах; random36 ухудшил attention error.

Лучший exploratory D4 - transpose: aggregate attention relative L2 0.0767430->0.0745026 (-2.9193%), tensor relative L2 улучшилась лишь на 0.1247%. Улучшились 19/24 пары и все четыре capture aggregates. Этот выбор сделан **после просмотра старых данных**.

Затем зафиксированы четыре новых authored текста, первые 384 token IDs, hashes и единственный первичный критерий: aggregate PPL transpose/identity <=1.0. Выполнены все 30 слоёв, causal continuation 95 targets/document, 380 targets/method. В сравнении V-layout обе ветки имеют одинаковый FP16 K и FP16 tail; asymmetric U4 - отдельный baseline с иным сжатием K.

| Метод на новых текстах | PPL / FP16 | Логические KV bytes перед последним прогнозом |
|---|---:|---:|
| FP16 | 1.000000 | 8 824 320 |
| Identity ASTC V-only | 1.013279 | 6 370 560 |
| Transpose ASTC V-only | 1.014430 | 6 370 560 |
| Asymmetric U4 K/V | 1.002874 | 4 020 480 |

**Первичный критерий FAIL:** transpose/identity=1.0011360377. Три документа хуже, один лучше. Это не доказательство статистически устойчивого вреда от transpose; это отсутствие заявленного подтверждения улучшения. Identity не заменён. Структура ошибки attention оказалась интересным диагностическим наблюдением, но её локальное улучшение не перенеслось в PPL этого теста.

Независимо пересчитаны все 1520 losses, target positions, ratios, память и hashes: PASS. Подробности: [геометрия](block_geometry_v2.md), [новые тексты и PPL](geometry_holdout_v2.md). Raw - `results/block_geometry_v2/`, `results/geometry_holdout_v2/`.

## 4. Настоящий fused baseline и исправление precision-контракта

Использован неизменённый llama.cpp commit `c3df4b7e22a1ac38da663552d19e649dd97ca936`, официальный `ggml_flash_attn_ext`, Metal backend, D64/Hq9/Hkv3. Запущены F16/F16 и Q4_0/Q4_0; mixed K/V в этой ревизии явно unsupported, fallback не выполняется. Q4_0 содержит 4.5 bits/scalar. Библиотека скомпилирована существующими CLT/CMake с runtime compilation MSL.

Первый запуск **не выдал timings**: gate `relative L2<=1e-3 AND max_abs<=1e-3*max(1,reference_max_abs)` провалился по max_abs. Причина оказалась в reference: официальный Metal kernel преобразует Q и Q4 dequant в half4. Отдельный precision probe сохранил GPU output и три независимых reference. После учёта этих округлений residual relative L2: F16 2.016e-7, Q4 2.704e-7. Предсказанный эффект округления почти полностью объяснил исходную ошибку. Это исправление тестового контракта, **не найденный баг llama.cpp**. Порог не ослаблялся.

Для новой серии Q заранее округлён FP16->FP32 **одинаково для ASTC и llama.cpp**; старый baseline executable и его строгий reference/gate сохранены. F16 K/V базы побайтно сверены с hardware ASTC readback, округлённым в F16. Q4 повторно квантует эту базу, поэтому серия сравнивает исполнение и корректность, а не исходное качество двух codec policies. Девять llama process runs прошли; max relative L2 F16 2.02e-7, Q4 5.84e-4. Во время timing не компилировались новые pipelines.

| T | ASTC v4, host sync us | llama F16, host sync us | llama Q4_0, host sync us |
|---:|---:|---:|---:|
| 384 | 156.062 | 155.833 | 160.250 |
| 2048 | 164.041 | 155.916 | 160.333 |
| 8192 | 188.042 | 167.834 | 175.771 |

Три process sessions, 150 rounds/variant; в таблице медиана трёх session medians. Одна host submission до завершения GPU, output readback вне timing. Разница границ framework/dispatch остаётся внутри wall time. Сеансовая вариативность заметна: например Q4/T384=147.312-163.562 us, ASTC=153.354-160.979 us. Устойчивое преимущество ASTC не установлено; эти host numbers нельзя сравнивать с batched GPU us из раздела 2 как одну метрику.

Память тоже требует одинаковой границы. Для T384 логические Q4 KV - 82 944 bytes, но официальный backend резервирует 472 320 bytes вместе с tensor/scratch allocation. ASTC V-only persistent K+V+metadata - 201 216 bytes, собственный attention scratch при tile32 - 35 936 bytes. Это разные политики сжатия и разные runtime reserves; приведённые числа не доказывают равенство качества или общий peak-memory выигрыш. Диагностические процессы держат дополнительные controls живыми.

Первичные источники и детали: [baseline, build, precision](llama_baseline_v2.md), [pinned fa_vec source](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_common.metal). Исходный FAIL, GPU diagnostic и успешная новая серия сохранены отдельно: `results/kernel_v2/llama_comparison/`, `llama_precision.json`, `llama_precision/`, `llama_comparison_common_q/`.

## 5. Проверка большого рабочего набора

Протокол зафиксирован в `configs/kernel_pool_v5.json` до измерений: T8192/tile256, отдельные resident allocations, N=1/8/64, три сеанса, 30 batches по одному полному циклу пула. Primary N64, positive: ASTC >=20% быстрее собственного FP16 во всех сеансах; negative: ASTC >5% медленнее во всех сеансах; иначе inconclusive. Общий cap pool allocation 1 GiB. Первоначальное предложение независимого рецензента N128 было конкретизировано координатором до измерений; неисполнявшийся вариант не является протоколом этого запуска.

| Cache slots | ASTC GPU us/call | FP16 GPU us/call | ASTC read-resource allocation | FP16 read-resource allocation |
|---:|---:|---:|---:|---:|
| 1 | 43.000 | 27.146 | 4 816 896 B | 6 291 456 B |
| 8 | 45.289 | 34.063 | 38 535 168 B | 50 331 648 B |
| 64 | 44.880 | 34.017 | 308 281 344 B | 402 653 184 B |

Медианы трёх session medians. **Предзаданный результат NEGATIVE:** ASTC/FP16 при N64 = 1.321993, 1.319352, 1.318586. Большой набор уменьшил относительный проигрыш, но не изменил его знак. Все данные предсжаты; ещё отсутствующие encoding/update расходы не включены. Это NO-GO для дорогой реализации активного runtime под проверенный scalar6/token/identity/D64 путь с прежним 3-slice/66-wide атласом без нового технического объяснения; не универсальное опровержение ASTC.

В каждом slot разные K/V/metadata allocations; K этого slot намеренно общий двум сравниваемым веткам. Все диагностические controls и оба V-представления одновременно resident, всего pool 509 607 936 B при N64. В таблице отдельно посчитаны ресурсы, которые читает каждый shader; это не process peak. Нет аппаратных counters/cache-miss evidence, поэтому результат не называется измерением cold-cache DRAM bandwidth.

Все 219 slot hash sets, every-slot outputs и wraparound прошли. Независимый post-run audit проверил runtime hashes Q, K, V, metadata и ASTC payload во всех девяти сеансах, соответствие исходным fixtures и отдельному ранее полученному hardware readback V, циклы курсора, memory arithmetic и raw medians. Этот audit закрыл отмеченный review пробел: Python runner сам не включал data files в свой начальный watched list, но необходимые фактические input signatures сохранил GPU host. Прежние результаты не переписывались. Проверка: `scripts/verify_pool_results.py`; raw: `results/kernel_v2/pool_v5/`; [дизайн и ограничения](pool_probe_v5.md).

### 5.1. Новый проверенный результат: физическая упаковка без изменения блоков

После negative pool gate выявлено конкретное новое объяснение расхода памяти: на T8192 полезная ASTC payload равна 726 528 B, а прежняя V texture реально выделяет 1 572 864 B. Этот факт мотивировал **отдельный** allocation-only опыт с 18 shapes; исходный negative gate не изменялся. Compact atlas шириной 32 ASTC-блока уменьшил выделение до 737 280 B. В нём исходные блоки последовательно размещаются head->block row->block column; сами 16-byte blocks не меняются. На T384 выигрыш allocation отсутствует. Это перенос целых блоков, а не неудавшийся transpose значений внутри блока.

v6 затем реализовал настоящий upload, адресацию каждого scalar через 32-block-wide atlas, независимый scalar readback и восьмую attention-ветку. Source width/height и metadata остаются исходными; decoder использует padded source height. До timing требуется **побитное совпадение всех normalized V float32** с исходным аппаратным decode. Все девять основных process runs и отдельный T391/tile13 контроль прошли; последний проверяет неполную страницу, wave и последнюю atlas row с 256 B valid padding. Max attention error против того же FP64 reference - 9.31e-7.

| T | Прежняя V texture, B | Compact V texture, B | Прежние K+V+metadata, B | Compact K+V+metadata, B | GPU penalty compact/original, matched-round medians по сеансам |
|---:|---:|---:|---:|---:|---|
| 384 | 49 152 | 49 152 | 201 216 | 201 216 | +6.83 / +6.98 / +6.85% |
| 2048, повтор блоков | 393 216 | 196 608 | 1 204 224 | 1 007 616 | +9.81 / +9.06 / +9.73% |
| 8192, повтор блоков | 1 572 864 | 737 280 | 4 816 896 | 3 981 312 | +14.37 / +14.47 / +14.58% |

На T8192 это -17.35% persistent K+V+metadata относительно прежнего ASTC layout, либо -36.72% относительно собственного двухстороннего FP16 buffer (6 291 456 B). Это **не** -53% всей KV-памяти и не process peak. Staging/upload, query/scratch и дополнительные диагностические controls учитываются отдельно в JSON. Предварительная перепаковка CPU и upload исключены из attention timing; online append/reallocation для новой раскладки не реализованы.

Стабильный T8192 контроль: original ASTC 38.673 us, compact 44.271 us, собственный FP16 25.525 us. На T384 абсолютные session medians заметно дрейфовали у всех веток (original ASTC 7.965 / 11.565 / 25.000 us); причина не установлена, все сеансы сохранены. Поэтому для эффекта layout указаны отношения внутри одинаковых shuffled rounds, а не выбранный самый быстрый сеанс. Smaller allocation не превратился в speedup в этой серии. Compact вариант также не прошёл бы условие "память лучше при latency penalty<=5%" относительно собственного FP16 в этом microbenchmark; полный runtime этим опытом не оценивается.

Выбор width32 сделан по завершённому allocation sweep до первого compact attention timing. Это exploratory co-design и проверенная инженерная возможность, не доказанная научная новизна. Она даёт конкретный новый объект дальнейшего исследования - цену адресации/упаковки при тех же представленных значениях - и не отменяет negative outcome старого layout. Источники: `configs/kernel_atlas_v6.json`, `kernel_atlas_ragged_v6.json`, `results/kernel_v2/atlas_allocation.json`, `atlas/`, `atlas_ragged/`, `derived_atlas_summary.json`; [контракт v6](atlas_attention_v6.md).

## 6. Новизна и границы вывода

SIMD scheduling, split softmax, перестановки block layout и algebraic affine factoring сами по себе не являются доказательством новизны. Квантование внимания и организация вычислений вокруг quantization scales уже исследуются, например в [SageAttention](https://openreview.net/pdf?id=OL44KtasKc); ASTC weights и другие compressed-KV направления разобраны в [prior art](independent_research_20261028.md). Нового доказанного ASTC-specific полезного принципа текущая серия не установила.

Работа остаётся precompressed attention probe на одной форме/модели. GPU encoder, append/update, FP16 tail merge в рабочем runtime, полный model decode latency и вторая модель не реализованы/не измерены этой серией. CPU PPL опыт проверяет однократное сжатие старого prefix с последующим FP16 tail и реальным feedback модели. Отрицательные результаты относятся к конкретным проверенным режимам; они не опровергают все будущие ASTC представления.

## 7. Выполненная верификация и воспроизведение

Все команды выполняются из корня `astc-kv`; повторные experiments требуют **новых** output directories. [Точные выполненные команды с фактическими output paths и статусами](../results/kernel_v2/executed_commands.txt) сохранены отдельно; ниже приведены рецепты повторения с `NEW_*` placeholders. Metal требует разрешённого доступа к GPU вне текущей песочницы. Наличие исходников не заменяет runtime compilation: здесь она реально состоялась.

```sh
bash metal/build_v2.sh
bash metal/build_v3.sh
bash metal/build_v4.sh
.venv-research/bin/python scripts/run_kernel_sweep.py --fixtures results/kernel_v2/fixtures_page --out NEW_SPLIT
.venv-research/bin/python scripts/run_kernel_sweep.py --config configs/kernel_attribution_v2.json --binary build/metal-v2/astc-kv-probe-v2 --shader metal/attention_v2.metal --fixtures results/kernel_v2/fixtures_page --out NEW_ATTRIBUTION
.venv-research/bin/python scripts/run_kernel_sweep.py --config configs/kernel_factored_v3.json --binary build/metal-v3/astc-kv-probe-v3 --shader metal/attention_v3.metal --fixtures results/kernel_v2/fixtures_page --out NEW_FACTORED
.venv-research/bin/python scripts/run_kernel_sweep.py --config configs/kernel_wave_v4.json --binary build/metal-v4/astc-kv-probe-v4 --shader metal/attention_v4.metal --fixtures results/kernel_v2/fixtures_page --out NEW_WAVE
.venv-research/bin/python scripts/verify_metal_v3.py --out-dir NEW_V3_CONTROLS
.venv-research/bin/python scripts/verify_metal_v4.py --out-dir NEW_V4_CONTROLS
bash scripts/build_llama_baseline.sh
build/llama-baseline/bin/astc-llama-baseline --cpu-self-test
build/llama-baseline/bin/astc-llama-precision results/kernel_v2/llama_comparison/matched_inputs/t384/manifest.json NEW_PRECISION_DIR
.venv-research/bin/python scripts/compare_llama_metal_common_q.py --out NEW_COMMON_Q
bash metal/build_pool_v5.sh
.venv-research/bin/python scripts/run_pool_sweep.py --out NEW_POOL
.venv-research/bin/python scripts/verify_pool_results.py
bash metal/build_atlas_alloc.sh
build/metal-atlas-alloc/astc-kv-atlas-alloc
bash metal/build_atlas_v6.sh
.venv-research/bin/python scripts/run_kernel_sweep.py --config configs/kernel_atlas_ragged_v6.json --binary build/metal-atlas-v6/astc-kv-probe-atlas-v6 --shader metal/attention_atlas_v6.metal --fixtures results/kernel_v2/fixtures_page --out NEW_ATLAS_RAGGED
.venv-research/bin/python scripts/run_kernel_sweep.py --config configs/kernel_atlas_v6.json --binary build/metal-atlas-v6/astc-kv-probe-atlas-v6 --shader metal/attention_atlas_v6.metal --fixtures results/kernel_v2/fixtures_page --out NEW_ATLAS
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-research/bin/python -m unittest discover -s tests -v
.venv-research/bin/python scripts/summarize_kernel_research.py --verify
.venv-research/bin/python results/block_geometry_v2/verify_saved.py
.venv-research/bin/python results/geometry_holdout_v2/verify_saved.py
git diff --check
.venv-research/bin/python scripts/freeze_research_v2.py --verify
```

PASS: host builds, real GPU gates/sweeps, common-Q baseline, 15 unit tests, raw numerical recomputation, whitespace check. Expected negative results retained: initial malformed ASTC-height fixture/old host encoder cleanup abort (v2 path repaired and refusal retested), initial baseline precision gate, transpose holdout criterion. SKIPPED/not performed: full dynamic runtime, production inference benchmark, second model, exhaustive numerical/stress coverage. CPU geometry/holdout exact run/freeze commands and hashes are in their separate reports.
