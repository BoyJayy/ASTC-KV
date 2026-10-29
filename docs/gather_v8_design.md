# Paired ASTC gather experiment v8

Статус подготовки: host build и shell syntax PASS. GPU compilation, gather semantics, correctness и timings агент подготовки не запускал. Root выполняет аппаратную серию централизованно. Старые frozen files и результаты не меняются.

V8 проверяет конкретную гипотезу: один gather возвращает четыре исходных scalar texels из 2x2 footprint, после чего weighted sum остается обычной FP32 арифметикой. Payload, decoded representation, compact atlas, Q/K, metadata и persistent memory сохраняются. Это не linear sampling и не новый codec.

## Почему выбран этот опыт

Ранний attribution показывает, что объяснение только количеством texture reads недостаточно. В v2 при T8192/tile128 normalized FP16 buffer+affine занимал 79.873 us, ordinary scalar texture 77.900 us, ordinary RGBA texture 78.256 us, ASTC 72.683 us и restored FP16 buffer 46.250 us (median session medians). В v4 при T8192/tile256 ASTC factored и normalized buffer factored были близки: 38.370 и 39.636 us против own FP16 25.393 us. Поэтому выигрыш gather не предполагается заранее.

Gather выбран как ограниченный опыт с неизменными значениями. RGBA ASTC потребовал бы нового encoding и отдельной оценки потерь. Linear sampling вводит precision interpolation weights и риск нарушения строгого gate. GQA reuse изменил бы совместное исполнение трех query heads и объем shader state; matrix-based split score/V - несколько kernels и промежуточную память. Эти направления здесь не реализуются.

## Геометрия и API

Новые файлы: `metal/attention_gather_v8.metal`, `metal/probe_gather_v8.mm`, `metal/build_gather_v8.sh`. Shader сначала содержит все неизмененные v7 functions, затем добавляет новый paired kernel и диагностическую проверку. Host по-прежнему требует D64, три KV heads, scalar ASTC6, source width66, FP16 K и token-normalized V.

Если token и channel четные, пара token/token+1 и channel/channel+1 не пересекает ASTC 6x6 block: остатки начальных координат равны 0, 2 или 4. Поэтому она также не пересекает remapped compact-atlas seam. Source padded height используется как прежде.

MSL gather требует `access::sample`, возвращает четыре компоненты выбранного канала и не интерполирует их. Порядок из MSL specification: `(-,+), (+,+), (+,-), (-,-)` вокруг центра footprint. Для integer texture coordinates это:

```text
g.x = U[token+1, channel]
g.y = U[token+1, channel+1]
g.z = U[token,   channel+1]
g.w = U[token,   channel]
```

Спецификация Apple доступна по [официальной ссылке MSL](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf); порядок и overload `texture2d_array.gather` также проверены по [архивному тексту спецификации Apple 2017](https://paperzz.com/doc/7365042/metal-shading-language-specification). Совместимость конкретного ASTC gather path остается аппаратным gate. [Apple feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) документируют ASTC sampling/filtering capability, но это не доказательство конкретного результата этого kernel.

Gather использует constexpr sampler с normalized coordinates и clamp-to-edge. Координата центра равна `(float2(x,y)+1)/texture_size`, где x/y - первый texel квадрата. Этот центр удален от границы выбора footprint. Тип texture - float; принудительного округления output в half нет. `filter::linear` в sampler описывает footprint gather, но weighted interpolation через `sample()` не выполняется.

## Paired kernel и fallback

QK и softmax по-прежнему вычисляются для восьми token quads. Каждый quad имеет четыре lanes и шестнадцать каналов на lane. Даже `token_lane` владеет своим token и следующим token. Weight, scale и offset соседнего quad получаются через `simd_shuffle_xor(...,4)`; shuffles выполняются всеми lanes, включая invalid tail.

Для пары:

```text
a = weight_even * scale_even
b = weight_odd  * scale_odd
value_channel_0 = a*g.w + b*g.x
value_channel_1 = a*g.z + b*g.y
offset_pair = weight_even*offset_even + weight_odd*offset_odd
```

Два gather покрывают четыре канала. Для шестнадцати каналов двух tokens это 8 gather calls вместо 32 scalar reads. Это число source-level запросов на пару; оно не означает 4x ускорение SIMD instructions, физического memory traffic или attention.

Нечетные token quads не загружают V и вносят ноль в pair reduction. Далее сохранена cross-quad reduction. Denominator суммирует все исходные token weights, включая оба tokens пары, ровно один раз через `quarter==0`. Старые state carry, merge и per-split output layout сохранены.

Если глобальный token пары нечетный или сосед отсутствует в текущем split, обе новые ветки используют отдельные mapped direct reads. Для отсутствующего соседа нет обращения к V, его weight/scale/offset равны нулю. Это обеспечивает tile1, tile7, tile13 и последний одиночный token. Source width/height и fragment coordinates не изменяются ради fallback.

## Варианты и gates

Все десять v7 variants сохранены как correctness controls. Добавлены:

| Index | Name | Function constant4 |
|---:|---|---|
| 10 | `compact_paired_direct_v` | false: четыре direct reads на 2x2 |
| 11 | `compact_paired_gather_v` | true: один gather на 2x2 |

Обе новые ветки имеют одинаковые pairing, arithmetic, fallback и schedule. Они используют тот же ASTC CPU FP64 reference и прежний gate: finite output, relative L2 <=1e-4. Pairing меняет порядок FP32 суммирования; gate не ослабляется.

Старый independent normalized decode остается bit-exact против original texture. Новый `check_compact_gather` для каждого head, token pair и channel pair сохраняет восемь float: native gather xyzw и четыре direct reads в ожидаемом порядке. Host требует finite и побитное совпадение через memcmp, записывает обе SHA256. Проверяются все четыре components, последний channel, все три heads и последняя строка; при нечетном T включен физически валидный padded neighbor.

JSON gate для runner:

```text
correctness.gather_vs_direct.passed_bit_exact_float32
correctness.gather_vs_direct.checked_pairs
correctness.gather_vs_direct.checked_scalar_values
correctness.gather_vs_direct.gather_sha256
correctness.gather_vs_direct.direct_sha256
```

Отдельный `compact_decode_vs_original_normalized` сохранен. Все двенадцать attention gates и обе exact проверки обязательны до timing. Failed gate печатает diagnostics с пустыми raw_samples и возвращает ошибку. Диагностический buffer отражен отдельно в `memory.gather_correctness_readback_allocated_bytes`; persistent K/V/metadata sums не изменяются.

## Timing contract

CLI как v7:

```text
astc-kv-probe-gather-v8 MANIFEST SHADER [blocks=30] [iterations=10]
    [tile_tokens=256] [readback_directory|-] [seed=20261028]
```

Сначала 10 warmup calls на каждый из двенадцати variants. Затем каждый round содержит три сравнения в перемешанном порядке, каждое - seeded ABBA или BAAB:

| pair_name | A | B |
|---|---:|---:|
| `hoisted_to_gather` | 8 | 11 |
| `paired_direct_to_gather` | 10 | 11 |
| `fp16_to_gather` | 2 | 11 |

Каждый sample сохраняет round, pair_name, slot_in_block, a_variant, b_variant, block_order, iterations и GPU/wall batch durations. Runner вычисляет `sqrt(B1*B2/(A1*A2))`. Correctness, decode, upload, SHA и pipeline creation вне timing. Host metadata pipeline сохраняется как слабая информация compiler/runtime, без register/occupancy claims.

Root-owned config `configs/gather_v8_20261029.json` задает 5 sessions, 40 blocks, 20 iterations, T384/tile32, T2048/tile64, T8192/tile256. Primary требует gather/hoisted <=0.85 в каждом из пяти T8192 sessions. Secondary paired-direct comparison отделяет gather от общего изменения paired schedule; FP16 comparison показывает оставшийся разрыв. Неуспех primary сохраняется без изменения порога.

## Сборка и передача

```bash
bash -n metal/build_gather_v8.sh
bash metal/build_gather_v8.sh
```

Обе команды PASS, host собран без warnings. Offline Metal compiler отсутствует; runtime shader compilation и GPU semantics еще не проверены этим агентом. Root подготовил независимые CPU geometry/pair algebra tests в `tests/test_gather_pairs_v8.py`; они не заменяют gather-versus-read hardware gate.

Короткий root smoke:

```bash
build/metal-gather-v8/astc-kv-probe-gather-v8 \
  results/kernel_v2/fixtures_page/t391/manifest.json \
  metal/attention_gather_v8.metal 1 1 13 - 20261031
```

Зарегистрированная серия:

```bash
.venv-research/bin/python scripts/run_gather_v8.py --out results/research_v4_20261029/gather_main
```

Output directory должен быть новым. Root сохраняет hashes и все исходные результаты. Старые frozen shader/host/runner files этим изменением не затрагиваются. До аппаратных gates и измерений не заявляются exact gather PASS или ускорение.
