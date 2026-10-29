# Compact atlas address experiment v7

Статус подготовки: host build и CPU address simulation PASS. GPU compilation, correctness и timing агент подготовки не выполнял. Исходные frozen v6 files и результаты не изменялись.

Цель: измерить влияние формы вычисления адресов на том же compact atlas. В v6 каждый scalar read описан через деление и остаток от деления координат. V7 сохраняет этот вариант и добавляет два. Не предполагается, что compiler обязательно исполняет все операции исходного текста: он мог уже выполнить часть выноса вычислений. Доказательство ускорения требует аппаратных результатов, включая отрицательные исходы.

Новые файлы: `metal/attention_atlas_v7.metal`, `metal/probe_atlas_v7.mm`, `metal/build_atlas_v7.sh`, этот документ и `build/metal-atlas-v7`. Ограничения прежние: D64, scalar ASTC 6x6 LDR, три KV heads, source width66, FP16 K, token-normalized ASTC V. Source height остается padded height исходной texture. Все новые пути factored и используют тот же compact texture, Q, metadata, payload, CPU reference и scratch.

## Три режима адресования

Function constant3 имеет тип uint. Host явно передает 0 для всех прежних pipelines и задает 1 или 2 только двум новым pipelines.

| Index | Variant | Mode | Отличие |
|---:|---|---:|---|
| 7 | `astc_compact_factored_v` | 0 | Прежний v6 helper без изменения |
| 8 | `astc_compact_hoisted_v` | 1 | Channel maps и head base вычисляются до token loop |
| 9 | `astc_compact_recurrent_v` | 2 | Дополнительно обновляет token block row и remainder без div/rem внутри loop |

Для mode1 и mode2 четыре `uint4` хранят `c/6`, еще четыре хранят `c%6` для шестнадцати каналов lane. Они создаются один раз до token loop. `headBase = kv_head*(source_height/6)*11` тоже вынесен из loop. Каждая scalar компонента сохраняет собственный адрес, в том числе на 6-channel block boundary и 32-block atlas-row boundary.

Mode1 для каждой wave и lane вычисляет `tokenBase = headBase + (t/6)*11`, `withinY = t%6`. Затем mapped helper использует:

```text
block = tokenBase + channelBlock
x = (block & 31)*6 + channelWithin
y = (block >> 5)*6 + withinY
```

Mode2 инициализирует base и remainder для `t = split_start + token_lane` один раз. При переходе к следующей wave t увеличивается на 8. Если `t = 6*q+r`, `0 <= r < 6`, то `t+8 = 6*(q+1)+(r+2)`. Поэтому обновление:

```text
carry = r >= 4
tokenBase += 11*(1+carry)
r = carry ? r-4 : r+2
```

Оно выполняется всеми lanes после каждой wave, включая invalid lanes последней wave. Чтения texture и metadata по-прежнему находятся под `valid` guard. Нет изменения QK, softmax, factored V math, split/merge, Q или metadata.

Mode1 против mode0 показывает суммарный эффект новой формы адресования при том же порядке texture reads. Mode2 против mode1 проверяет дополнительный эффект recurrence. Это не прямое измерение цены одной div instruction: compiler scheduling, register pressure и зависимые операции также могут измениться. Раскладка texture и логическая последовательность адресов одинаковы, но без hardware counters нельзя объявлять доказанную причину изменения latency.

## Correctness и измерения

Сохранены восемь прежних вариантов v6 и все их gates. Добавлены две own-reference проверки новых attention paths: finite output и relative L2 <= 1e-4 против той же ASTC CPU FP64 accumulation reference. Порог не меняется.

CPU repack и payload hashes не изменены. Старый независимый `decode_compact_texture` тоже не изменен: он требует побитного совпадения decoded normalized FP32 V с исходной texture до timings. Новый mapped helper этот decoder не использует; ошибка helper должна проявиться в attention gate, а не скрыться общей реализацией reference.

Десять вариантов перемешиваются seeded shuffle в каждом warmup и measured round. Сохраняются 10 warmup calls на вариант, defaults 30 rounds и 10 iterations, тот же batch timing и raw samples. Compilation, repack, upload, decode и reference находятся вне attention timing. Память compact representation не изменяется; дополнительные channel maps - shader-local state, размер register allocation этим probe не измеряется.

В `pipeline_metadata` на каждый variant, merge и decode записываются публичные свойства Metal pipeline:

- `thread_execution_width`;
- `max_total_threads_per_threadgroup`;
- `static_threadgroup_memory_length`.

Это ограниченная информация compiler/runtime. Она не измеряет occupancy, registers, instruction count или cache misses. Profiler не запускался.

## Проверки подготовки

```bash
bash -n metal/build_atlas_v7.sh
bash metal/build_atlas_v7.sh
xcrun clang++ -std=c++17 -O2 -Wall -Wextra \
  build/metal-atlas-v7/address_sim.cpp -o build/metal-atlas-v7/address-sim
build/metal-atlas-v7/address-sim
```

Все команды PASS. Host и CPU simulation собираются без предупреждений существующим CLT. Offline `metal` compiler отсутствует; shader syntax и GPU execution остаются непроверенными до root runtime run.

CPU simulation сравнивает mode0, mode1 и mode2, затем восстанавливает исходные head/token/channel из compact coordinates. Проверены 221 shape/tile cases: T1/5/6/7/8/13/31/32/33/95/96/97/383/384/391/2048/8192 и tiles1/5/6/7/8/13/16/31/32/64/127/256/513. Включен T391/tile13, head/page/row boundaries, полный набор каналов и inactive final lanes.

Результат: 48580608 scalar comparisons, 29507712 valid coordinates, 298014 invalid lane waves и 759072 recurrence steps, включая последнее обновление. Сохранено в `build/metal-atlas-v7/address_validation.json`. Это проверка арифметики на CPU, не замена GPU gate. При self-review исходные merge/decode kernels, v6 compact helper, repack и value/decode gate blocks сравнены без изменений.

## Передача аппаратного запуска

Короткий ragged check для основного агента:

```bash
build/metal-atlas-v7/astc-kv-probe-atlas-v7 \
  results/kernel_v2/fixtures_page/t391/manifest.json \
  metal/attention_atlas_v7.metal 1 1 13 - 20261029
```

Обычный T8192 batch с теми же default rounds/iterations:

```bash
build/metal-atlas-v7/astc-kv-probe-atlas-v7 \
  results/kernel_v2/fixtures_page/t8192/manifest.json \
  metal/attention_atlas_v7.metal 30 10 256 - 20261029
```

Вывод направлять только в новые файлы. Root запускает GPU централизованно и фиксирует session schedule перед измерениями. Exact source/binary hashes и build commands находятся в `build/metal-atlas-v7/preparation.json`. До аппаратных данных не заявляются GPU PASS, speedup или устранение bottleneck.
