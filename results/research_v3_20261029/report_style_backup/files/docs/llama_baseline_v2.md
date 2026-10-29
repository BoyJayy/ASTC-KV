# Standalone fused Metal baseline: pinned llama.cpp

Статус: официальный ggml и host собраны, CPU glue/GQA self-test выполнен. **Реальные fused Metal F16/Q4_0 pipelines выполнены на T384. Первоначальный strict gate не прошёл из-за различия точности operand/reference; отдельная GPU-диагностика подтвердила механизм и дала residual relative L2 около 2e-7.** Исходный провал сохранён, benchmark source и gates не изменены. Серия timings на общем заранее округлённом Q завершена: девять официальных Metal process runs, все gates PASS, новых pipeline compilations в timing нет. Raw: `results/kernel_v2/llama_comparison_common_q/`; сводка и сопоставимые границы измерения — в [общем отчёте](research_v2_report.md). Max relative L2 F16=2.02e−7, Q4=5.84e−4; исходный strict gate и executable не изменены.

## Источник и граница реализации

Используется неизменённый публичный [llama.cpp `c3df4b7e22a1ac38da663552d19e649dd97ca936`](https://github.com/ggml-org/llama.cpp/tree/c3df4b7e22a1ac38da663552d19e649dd97ca936), скачанный в `third_party/llama.cpp-c3df4b7e22a1ac38da663552d19e649dd97ca936`. Archive SHA256: `d82221b4491ba3879ec55786eaf0cc7cae468b05951b12e508e8c614b49f4374`; download provenance — `third_party/llama_baseline_provenance.json`.

Новые файлы только в `baselines/llama/`, `scripts/build_llama_baseline.sh`, `build/llama-baseline/` и этот документ. Ядра ASTC v1/v2/v3 и upstream source не меняются. Harness вызывает официальный `ggml_flash_attn_ext` непосредственно на выбранном backend. Graph обязан содержать ровно один `GGML_OP_FLASH_ATTN_EXT`; scheduler и CPU fallback не используются. Поддержка операции проверяется до выделения и запуска.

Формы ggml, от самого быстрого измерения к медленному:

- Q: `[D,1,Hq,1]`, всегда один и тот же FP32 input;
- K/V: `[D,T,Hkv,1]`, каждый токен — contiguous row;
- output: `[D,Hq,1,1]`, flat-индексы совпадают с CPU reference;
- GQA `kv_head=floor(query_head/(Hq/Hkv))`, D64 либо D128;
- один causal decode query ко всему переданному допустимому префиксу; mask не нужен, prefill не реализован.

## Варианты и фактическая поддержка

| Вариант | K | V | Логические биты на KV-скаляр |
|---|---|---|---:|
| `f16_kv` | F16 | F16 | 16 |
| `q4_0_kv` | Q4_0 | Q4_0 | 4.5 |
| `q4_0_k_f16_v` | Q4_0 | F16 | не запускать, если backend отверг |
| `f16_k_q4_0_v` | F16 | Q4_0 | не запускать, если backend отверг |

В pinned `ggml-metal-device.m` поддержка `FLASH_ATTN_EXT` явно отклоняет `K.type != V.type`. Поэтому ожидается, что обе mixed branches вернут `unsupported_by_selected_backend`, `fallback_executed=false`. Это проверяется через публичный backend API при реальном запуске, а не имитируется успешной веткой. Upstream предоставляет fused F16 и Q4_0 kernels для D64/D128.

Q4_0 получается официальным `ggml_quantize_chunk`, затем независимо деквантован официальным `ggml_get_type_traits(type)->to_float` только для CPU reference вне timing. Группа — 32 значения: 16 payload bytes и 2-byte FP16 scale, всего 18 bytes / 32 values = 4.5 bits. Это официальный Q4_0, не наш прежний affine U4 и не KIVI.

Для batch1 D64/D128 `ggml_metal_op_flash_attn_ext_use_vec` выбирает vector kernel; `use_kv_f16` должен быть false, то есть полного предварительного GPU-деквантования K/V для этого dispatch нет. На T384/D64 фактически зафиксированы `kernel_flash_attn_ext_vec_f16_dk64_dv64`, `kernel_flash_attn_ext_vec_q4_0_dk64_dv64` и reduce kernel, официальный Metal backend и исполнение single-op graph. Обе mixed branches реально вернули unsupported без fallback. Это подтверждает выбранный путь на этой форме; D128 аппаратно этим опытом не проверен. Материализация CPU decode нужна только эталону.

## Какие значения сравниваются

Harness читает обычный `astc-kv-probe-v1` manifest и `query_file`. Для FP16 стороны используется её `k_file`/`v_file`; для ASTC стороны — сохранённый `k_decoded_file`/`v_decoded_file`, а не ASTC texture. Обе стороны приводятся к одной F16 базе **до** выбора F16 или Q4_0. Q не округляется дополнительно. После correctness-call содержимое каждого Q tensor читается обратно и обязано побитно совпасть с входом.

У сжатой стороны эта база по умолчанию уже содержит ошибку offline ASTC decode. Поэтому это сравнение выполнения на контролируемых реконструированных значениях, не доказательство качества Q4_0 на исходных K/V модели. Прямой аппаратный ASTC decode немного отличается от software decode; это ограничение должно оставаться видимым при сравнении с ASTC.

Для строго согласованного readback можно добавить в отдельную копию manifest поля `baseline_k_f32_file` и/или `baseline_v_f32_file`. Они имеют приоритет и задают относительный путь к явно выбранному F32 `[Hkv,T,D]`; harness округлит его в F16 для обеих baseline-веток. Paths должны оставаться внутри каталога manifest. JSON сохраняет выбранный источник каждой стороны, SHA256 manifest/Q/source bytes/base F16 и packed payload каждого варианта. Для model-quality baseline нужны исходные настоящие K/V и отдельная полная оценка качества.

## Correctness gate до timing

Для каждого реально поддержанного варианта собственный CPU reference использует:

1. Тот же неизменённый FP32 Q.
2. Значения, восстановленные **из реально загруженных packed bytes** официальным decoder.
3. QK dot products, stable softmax и weighted V в FP64, затем FP32 output.

Условие, выбранное до первого GPU-прогона этого harness: все значения конечны, relative L2 ≤ 1e-3 **и** `max_abs <= 1e-3*max(1, reference_max_abs)`. Это engineering gate точности исполнения на одинаковых представленных значениях, не бюджет потерь кодека. Порог учитывает, что официальные Metal kernels могут использовать half intermediates даже при FP32 query/input и precision request. При провале нет warmup/timing: JSON содержит `correctness_failed`, reference errors и пустые `raw_samples`; exit 2. Порог не изменяется автоматически по результату.

Не использовать CPU attention ggml как независимый эталон: исходный CPU self-test с дробными постоянными V выявил F16 relative L2 `0.004502653146731495`, max_abs `0.036458492279052734`, тогда как Q4_0 совпал точно. В pinned `ggml/src/ggml-cpu/ops.cpp` V типа F16 накапливается в `VKQ16` через `ggml_vec_mad_f16`, включая dispatch при `GGML_PREC_F32`. Первичный провал сохранён в `build/llama-baseline/cpu-fractional-f16-accumulation-failure.json`; это отдельный CPU-результат, не GPU-ошибка.

Итоговый ограниченный glue test использует zero Q, T96, Hq9/Hkv3, D64 и V, кратные 0.5: все частичные суммы остаются точно представимыми в F16. Ожидаемый результат дополнительно проверяется прямым средним V для каждой GQA-группы. F16 и Q4_0 совпали точно с этим независимым oracle (`max_abs=0`, relative L2=0). Это проверка формы, группировки и кодирования, не сертификат точности CPU kernel на реальных входах.

## Выполненная GPU-диагностика контракта точности

Первый аппаратный запуск на согласованных hardware readback K/V, T384/D64/Hq9/Hkv3, сохранён в [исходном результате](../results/kernel_v2/llama_comparison/s0_llama_t384_tile0.json) и соответствующем `.stderr`. Обе ветки имели relative L2 меньше 1e-3, но превысили заранее заданный max-absolute gate. Поэтому timings не выполнялись и не публиковались:

| Ветка | Relative L2 против первоначального reference A | Max absolute error | Итог исходного gate |
|---|---:|---:|---|
| F16 | 0.0008267093954065612 | 0.0017875134944915771 | FAIL |
| Q4_0 | 0.0008789822710637614 | 0.0017895549535751343 | FAIL |

Этот результат сам по себе **не указывает на upstream bug**. Независимый разбор обнаружил различие исходного reference и фактических операндов kernel:

- В [F16 `FA_TYPES`](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_f16.metal#L5) и [Q4_0 `FA_TYPES`](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_q4_0.metal#L5) типы Q/K/V — `half4`, а QK, softmax и output accumulation — float/float4.
- [Загрузка Q в shared memory](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_common.metal#L110) явно выполняет `(q4_t) q4[i]`, то есть округляет FP32 query в half. Входной Q buffer при этом остаётся неизменным; проверка отсутствия мутации корректна, но не означает отсутствие внутреннего округления.
- [Q4_0 decoder](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/dequantize.h#L151) записывает деквантованные значения в half4 registers. Прежний официальный CPU `to_float` выдаёт FP32 значения, поэтому первоначальный reference также пропускал это округление.
- [QK dot products](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_common.metal#L275), [online softmax](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_common.metal#L401), [weighted V](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_vec_common.metal#L428) и [финальное merge](https://github.com/ggml-org/llama.cpp/blob/c3df4b7e22a1ac38da663552d19e649dd97ca936/ggml/src/ggml-metal/kernels/fa_aux.metal#L290) выполняются в FP32. Здесь не подтверждается гипотеза общего half-накопления GPU attention.

Эти первичные исходники прочитаны из локальной копии проверенного pinned-архива; ссылки закреплены на той же ревизии. Диагностический файл `baselines/llama/precision_probe.mm` отдельно использует прежний harness и сохраняет actual GPU outputs и три reference: A = исходный FP32 Q + CPU decode; B = half-rounded Q + CPU decode; C = half-rounded Q + half-rounded decoded K/V. Ни исходный `probe.mm`, ни его gate не переписывались.

CPU-предсказание до диагностического GPU run почти точно воспроизвело наблюдаемую величину ошибки. Затем [аппаратная диагностика](../results/kernel_v2/llama_precision.json) подтвердила механизм:

| Ветка | Actual против B: relative L2 | Actual против C: relative L2 | Actual против C: max abs |
|---|---:|---:|---:|
| F16 | 2.016344618123987e-7 | 2.016344618123987e-7 | 3.5762786865234375e-7 |
| Q4_0 | 0.0005623322974867832 | 2.7037673589391943e-7 | 4.76837158203125e-7 |

Reference C проходит **тот же численный порог**, без его ослабления. Cosine между фактической и предсказанной ошибкой округления: 0.9999999710 (F16), 0.9999999569 (Q4_0). SHA256 actual GPU outputs совпали с первоначальным запуском: диагностическая программа не изменила вычисление backend. Это сильное объяснение наблюдённого fixture failure через контракт точности, а не исправление upstream kernel.

Следующая серия явно меняет общий вход: Q заранее округляется F16 → F32 одинаково для ASTC и llama.cpp. Основной агент сохраняет первоначальный benchmark/reference/gate и пробует пройти его на трёх длинах с этим общим Q; ошибка Q4_0 half-dequant относительно CPU float decode остаётся отдельно видимой. Это новый сравнимый эксперимент, не переименование исходного FAIL в PASS. На момент этого дополнения серия common-Q timings запускается; результатов или ускорения здесь ещё не заявляем.

## Время и память

После correctness gate — десять отдельных warmup calls на вариант, затем по умолчанию 30 раундов. В каждом warmup и измеряемом раунде порядок поддержанных вариантов перемешивается с сохранённым seed. Сырые строки содержат variant, round, order и `host_sync_ms`.

Граница одного измерения: `steady_clock` вокруг одного `ggml_backend_graph_compute` и `ggml_backend_synchronize`. Подготовка, quantization, pipeline compilation, reference и output readback вне этого интервала. GPU-only time не измеряется и записан `null`. Сравнивать с ASTC можно только при той же границе одного host call с завершением работы; **нельзя делить batched GPU time ASTC на этот host latency**. Повторные чтения маленького resident fixture не дают long-context bandwidth или model speedup.

Сообщения компиляции fused pipelines сохраняются для каждого варианта. Если новая pipeline-компиляция обнаружена внутри timing, выставляется `timing_valid=false`, exit 3. Это не устраняет тепловой дрейф и host noise; нужны независимые сеансы.

Отдельно выводятся logical KV bytes и `backend_buffer_bytes_with_graph_scratch`. В pinned Metal backend helper `ggml_metal_op_flash_attn_ext_extra_kv_f16` **всегда резервирует F16 K/V workspace**, даже когда `use_kv_f16=false`; условный ранний выход закомментирован с пояснением об избежании reallocations. Следовательно, compressed direct read не означает, что выделение backend равно только packed KV. Этот harness сохраняет исходное поведение upstream. Одновременно живут несколько вариантов и CPU references, поэтому process peak не является production peak.

## Сборка и запуск

```bash
CMAKE=/opt/homebrew/bin/cmake bash scripts/build_llama_baseline.sh
build/llama-baseline/bin/astc-llama-baseline --cpu-self-test \
  > build/llama-baseline/cpu-self-test.json \
  2> build/llama-baseline/cpu-self-test.stderr
```

Сборка использует существующий CLT, C/C++/Objective-C/Objective-C++, Release, static ggml, CPU+Metal, без OpenMP/BLAS и без установок. `GGML_METAL_EMBED_LIBRARY=ON` в этой ревизии встраивает исходный MSL текст и использует официальный runtime compilation; отдельная утилита `metal` не нужна. Математические compile options upstream не заменяются. Host просит FP32 accumulation через `ggml_prec_set_acc`; реальная точность всё равно проверяется численно.

Первая сборка wrapper ошибочно включила OBJCXX без OBJC, из-за чего CMake классифицировал upstream `.m` как Objective-C++. Добавлено явное включение обоих языков; upstream не менялся. Первичный build failure сохранён в `build_attempt1_objc_language.log`. Окончательная сборка прошла без предупреждений в `build.log`.

GPU command для основного агента:

```bash
build/llama-baseline/bin/astc-llama-baseline \
  results/kernel_v2/fixtures_page/t384/manifest.json 30 20261028 \
  > NEW_OUTPUT.json 2> NEW_OUTPUT.stderr
```

CLI: `MANIFEST [rounds=30] [seed=20261028]` либо `--cpu-self-test`. Нужен разрешённый доступ процесса к GPU; агент подготовки его не запрашивал и GPU не запускал. Shell redirection не защищает существующий output: root выбирает новый путь результатов.

`build-provenance.json` содержит fixed full revision, проверенный archive hash, hashes wrapper/официальных kernels/quantizer/binary/CMakeCache/compile_commands и фактические compiler/SDK/macOS versions. CMake upstream ошибочно печатает `ggml commit: 869ffc2`, поскольку source archive лежит внутри родительского Git checkout; **это не идентификатор llama.cpp**. В итоговом JSON авторитетен явно закреплённый full `source_revision` и archive provenance.

## Выполненная проверка

| Команда/контроль | Статус |
|---|---|
| `bash -n scripts/build_llama_baseline.sh` | PASS |
| `CMAKE=/opt/homebrew/bin/cmake bash scripts/build_llama_baseline.sh` | PASS |
| `astc-llama-baseline --cpu-self-test` | PASS, обе представленные ветки exact |
| Дробный CPU F16 attention case | FAIL сохранён; объяснение FP16 V accumulation подтверждено исходником |
| GPU F16/Q4_0 dispatch, T384/D64 | Выполнен; обе официальные fused pipelines зафиксированы, mixed types отвергнуты без fallback |
| Первоначальный GPU gate против reference A | FAIL сохранён, timing withheld |
| Отдельная GPU-диагностика против reference C | PASS тем же порогом; residual relative L2 2.016e-7 / 2.704e-7 |
| 30-round host-latency benchmark на common-Q inputs | Запускается основным агентом; завершение и результаты здесь ещё не подтверждены |

Следующий шаг: основной агент завершает common-Q gate на каждой длине и сопоставимую paired host-latency серию с сохранением обоих численных контрактов. Только после этого можно оценить конкурентность ASTC относительно сильного fused baseline.
