# Gather exchange v9

Статус подготовки: host build, shell syntax и CPU route simulation PASS. GPU shader compilation, correctness и timings агент подготовки не запускал. Старые v7/v8 sources, binaries и результаты не меняются. Их hashes записаны до генерации и сборки v9 в `build/metal-gather-exchange-v9/frozen_inputs_before_build.json`.

## Проверяемая гипотеза

Предыдущий v8 run прошел correctness, но не достиг primary performance criterion. V8 объединял weighted contributions двух tokens после softmax. V9 сохраняет старый v7 порядок: normalized V загружается до weights, затем каждый token самостоятельно выполняет weighted accumulation. Gather используется только для совместной загрузки двух tokens; соседний quad получает свои данные через shuffle. Выигрыш не предполагается заранее.

Новые файлы: `metal/attention_gather_exchange_v9.metal`, `metal/probe_gather_exchange_v9.mm`, `metal/build_gather_exchange_v9.sh`, этот документ и `build/metal-gather-exchange-v9`. Весь shader v8 является неизмененным byte-identical prefix нового shader. Все двенадцать прежних variants и gates сохраняются.

## Новый kernel

Два варианта вызывают отдельный `attention_split_exchange`:

| Index | Name | Function constant5 |
|---:|---|---|
| 12 | `compact_exchange_direct_v` | false: четыре direct reads на 2x2 |
| 13 | `compact_exchange_gather_v` | true: один native gather на 2x2 |

Ограничения остаются прежними: D64, scalar ASTC6, source width66, три KV heads, FP16 K, token-normalized V. Q, compressed blocks, compact texture, metadata и CPU references одинаковы. Channel maps и `headBase` вынесены до token loop как в v7 hoisted. В каждой wave lane вычисляет свой `tokenBlockBase` и remainder.

Внутри прежнего цикла загрузки K/V каждая even token lane загружает 2x2 для своих и соседних normalized V values. Два таких footprint дают float4 собственного token и float4 соседнего token. Четыре float4 за цикл j покрывают шестнадцать каналов lane.

```text
pairToken = global_t - (token_lane & 1)
pairSafe = pairToken even and pairToken+1 < split_end
ownValues = (g0.w, g0.z, g1.w, g1.z)
oddValues = (g0.x, g0.y, g1.x, g1.y)
exchanged = simd_shuffle_xor(oddValues, 4)
```

Shuffle выполняется всеми 32 lanes без условия, включая invalid tail. Odd token lane использует `exchanged`, even lane - `ownValues`. Если начало пары глобально нечетное или сосед отсутствует, каждый действительный lane выполняет собственные mapped direct reads. Invalid lanes сохраняют нулевой V и не читают metadata/K/V. PairToken не предшествует split_start, поскольку пары образуются внутри текущей восьмитокенной wave.

После V loading идет прежний v7 блок QK reduction, online softmax, `v * (new_weight * scale)`, cross-quad value sum, offset sum, denominator/carry и store. Этот блок скопирован дословно. Нет pair weights, pair scale/offset arithmetic или переноса offset между tokens. Direct-exchange и gather-exchange отличаются только способом загрузки 2x2; schedule и shuffle одинаковы.

V9 по-прежнему меняет live shader state и добавляет shuffle относительно v7. Даже при улучшении нельзя объявлять доказанный register bottleneck или измеренную стоимость gather instruction без соответствующих аппаратных данных.

## Correctness и память

Сохраняются все v8 checks, включая `correctness.gather_vs_direct.passed_bit_exact_float32` и `compact_decode_vs_original_normalized`. Exact gather check уже отделен от attention и проверяет native components против четырех direct reads для всех heads/token/channel pairs.

Добавлены два attention gates против той же ASTC CPU FP64 accumulation reference: finite и relative L2 <=1e-4. Всего четырнадцать attention gates. Ни reference, ни численный порог не меняются. Failed gate прекращает работу до timing.

Persistent resources и allocation sums не меняются. Нет нового decoded V cache или дополнительного GPU staging в timed path. Остаются прежние diagnostic buffers и pipeline metadata. Shader-local temporaries не являются измеренным register count.

## ABBA contract

CLI сохраняет позиционные аргументы:

```text
astc-kv-probe-gather-exchange-v9 MANIFEST SHADER [blocks=30] [iterations=10]
    [tile_tokens=256] [readback_directory|-] [seed=20261028]
```

После десяти warmup calls на каждый из четырнадцати variants каждый round содержит четыре сравнения в seeded случайном порядке. Каждое comparison выбирает ABBA или BAAB:

| pair_name | A | B |
|---|---:|---:|
| `hoisted_to_exchange` | 8 | 13 |
| `exchange_direct_to_gather` | 12 | 13 |
| `paired_to_exchange` | 11 | 13 |
| `fp16_to_exchange` | 2 | 13 |

Raw samples содержат прежние `pair_name`, `round`, `slot_in_block`, `a_variant`, `b_variant`, `block_order`, `iterations`, GPU/wall batch durations. Ratio B/A рассчитывается как `sqrt(B1*B2/(A1*A2))`. Source-level request counts не равны измеренному physical traffic. ABBA не устраняет произвольное GPU interference; все blocks должны сохраняться.

## Проверки подготовки

```bash
bash -n metal/build_gather_exchange_v9.sh
bash metal/build_gather_exchange_v9.sh
xcrun clang++ -std=c++17 -O2 -Wall -Wextra \
  build/metal-gather-exchange-v9/exchange_sim.cpp \
  -o build/metal-gather-exchange-v9/exchange-sim
build/metal-gather-exchange-v9/exchange-sim
```

Все команды PASS, сборки без warnings. CPU simulation использует разные целочисленные labels для каждого head/token/channel и моделирует native gather ordering, even-owner values, xor4 transfer и fallback. Проверены 126 shape/tile cases, включая T391/tile13: 19545408 valid values, 6965568 odd-lane transfers, 5614272 fallback values и 22582464 нулевых invalid values. Результат сохранен в `exchange_validation.json`; это не аппаратная проверка shader.

Self-review проверяет: v8 shader prefix byte-identical, прежние repack/decoder/gather/value gates сохранены, v7 weight/softmax block дословно совпадает, ABBA mapping использует нужные indices. Offline Metal compiler отсутствует; новый shader будет собран runtime при root GPU run.

Root smoke command:

```bash
build/metal-gather-exchange-v9/astc-kv-probe-gather-exchange-v9 \
  results/kernel_v2/fixtures_page/t391/manifest.json \
  metal/attention_gather_exchange_v9.metal 1 1 13 - 20261031
```

Направлять output только в новые файлы. Root определяет и фиксирует experiment config до GPU timings. После freeze исходников не пересобирать binary во время серии. Подготовка, compiler и hashes сохраняются в `build/metal-gather-exchange-v9/preparation.json`. До аппаратных результатов не заявляются correctness PASS или speedup v9.
