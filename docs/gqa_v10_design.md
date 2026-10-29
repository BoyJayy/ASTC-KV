# GQA3 v10

V10 проверяет совместную загрузку K/V для трех query heads одной KV-группы. Это известный прием GQA reuse, а не заявление о новом алгоритме. Предыдущие gather experiments v8/v9 не достигли performance criterion. V10 меняет распределение работы между SIMD groups и не использует gather в двух новых путях.

Host build, shell syntax, CPU tests и GPU correctness PASS. Основная серия root завершена: primary performance criterion FAIL. Исходники и binary заморожены перед основной серией. Offline Metal compiler отсутствует; root успешно проверил runtime shader compilation аппаратными запусками.

## Файлы и сохраненные controls

Новые файлы: `metal/attention_gqa_v10.metal`, `metal/probe_gqa_v10.mm`, `metal/build_gqa_v10.sh`, этот документ и `build/metal-gqa-v10/`. Весь v9 shader сохранен как byte-identical prefix нового shader. Старые v7/v9 sources, build scripts и binary не менялись; hashes до сборки записаны в `build/metal-gqa-v10/frozen_inputs_before_build.json`.

Все четырнадцать старых variants и их references сохранены. Добавлены:

| Index | Name | Function constant 6 | CPU reference |
|---:|---|---|---|
| 14 | `compact_gqa3_astc_v` | true | Фактические hardware-decoded ASTC values с прежней affine reconstruction |
| 15 | `gqa3_fp16_buffer` | false | Фактические restored FP16 K/V values |

Scope: Hq=9, Hkv=3, D=64, scalar ASTC6, source width=66, FP16 K, token-normalized V. Остальные размеры и форматы отвергаются прежними validation checks либо новым Hq=9 check. Source padded height берется из manifest; compact texture и encoded blocks не меняются.

## Kernel и независимость query heads

Новый `attention_split_gqa3` запускается с `group.x=kh` и `group.y=split`. В каждой группе 32 lanes: четыре lanes на token и восемь tokens на wave. Каждый lane хранит три независимых набора `q`, `acc`, `m`, `denominator`, `offset_acc`, соответствующих `qh=kh*3+qi`, где `qi=0,1,2`.

Каждый channel group загружает один K float4 и один V float4. Эти значения используются для трех QK products и трех value accumulations. ASTC загружает normalized V и одну пару token scale/offset, затем для каждого query head выполняет прежнее factoring:

```text
sum(weight * (scale * U + offset))
  = sum((weight * scale) * U) + sum(weight * offset)
```

FP16 specialization загружает restored V из buffer; scale=1, offset=0. Она использует ту же GQA3 организацию работы, чтобы общий эффект reuse не приписывался одному ASTC.

Channel maps и head base вынесены до token loop как в v7 hoisted. Token block base и withinY вычисляются один раз на lane в wave. QK reduction использует xor1/xor2, V и offset reductions - xor4/xor8/xor16. Maximum, carry, numerator и denominator вычисляются отдельно для каждого query head. Invalid tail lanes не читают K/V/metadata, получают score=-INFINITY и нулевой contribution. Tile может быть нечетным или меньше восьми.

Число split groups по x сокращается с 9 до 3 только для indices 14/15. Запись partial/statistics сохраняет прежний индекс `state=qh*splits+split`. Прежний `attention_merge` запускается для всех девяти query heads. Новых threadgroup buffers, barriers или persistent decoded caches нет.

У трех query states больше live shader state, а SIMD groups меньше. Это может отменить пользу reuse. Доступные pipeline properties не измеряют register count, occupancy, spills или physical memory traffic, поэтому такие причины нельзя объявлять доказанными по одному timing result.

## Correctness и память

Всего шестнадцать own-reference attention gates: finite output и relative L2 <=1e-4 относительно CPU FP64 accumulation на фактически представленных values. Сохраняются независимые bit-exact checks `compact_decode_vs_original_normalized` и `gather_vs_direct`, а также прежние decode и ordinary/packed value checks. При любом failed gate timings не публикуются.

Persistent resource sizes и scratch layout не меняются. Для ASTC используется `compact_persistent_k_v_metadata_allocated_bytes`, для обоих FP16 schedules - `fp16_persistent_k_v_allocated_bytes`. Scratch и все diagnostic allocations отражены отдельно. Эти суммы не являются process peak: старые и новые controls, CPU references и textures существуют одновременно. Encoding, repacking, uploads и correctness readbacks находятся вне attention timing.

## CLI и ABBA

```text
astc-kv-probe-gqa-v10 MANIFEST SHADER [blocks=30] [iterations=10]
    [tile_tokens=256] [readback_directory|-] [seed=20261028]
```

После десяти warmup command buffers на каждый из шестнадцати variants round содержит четыре comparisons в seeded случайном порядке. Каждый comparison выполняет ABBA или BAAB с одинаковым iterations для A/B.

| pair_name | A | B |
|---|---:|---:|
| `hoisted_to_gqa` | 8 | 14 |
| `fused_fp16_to_astc` | 15 | 14 |
| `fp16_to_gqa_astc` | 2 | 14 |
| `fp16_to_gqa_fp16` | 2 | 15 |

Raw JSON сохраняет каждый `pair_name`, `round`, `slot_in_block`, `a_variant`, `b_variant`, `block_order`, `iterations`, GPU/wall batch durations. Ratio B/A вычисляется как `sqrt(B1*B2/(A1*A2))`. GPU batch duration и synchronized host duration остаются разными метриками.

Root заранее зафиксировал `configs/gqa_v10_20261029.json`: T=384/2048/8192, tiles=32/64/256, пять sessions, 40 blocks, 20 iterations. Primary для T8192 требует ASTC GQA3 <=1.05 от обоих FP16 controls в каждой session и persistent memory <=0.75 от FP16. Сравнение с прежним hoisted ASTC остается secondary. Ragged correctness cases: T391 с tiles 1,7,13,256. Пороги не изменяются по результатам.

## Проверки и запуск

Команды подготовки:

```bash
bash -n metal/build_gqa_v10.sh
bash metal/build_gqa_v10.sh
```

Обе PASS. C++ host собран существующим CLT с C++17/O3/ARC/Wall/Wextra. CPU state/protocol tests находятся в `tests/test_gqa_states_v10.py` и `tests/test_gqa_protocol_v10.py`; root сообщил пять tests PASS до основной GPU серии. Они проверяют три независимых softmax states, partial splits, extreme scores и отказ при некорректных timing labels/числах. Сам host/shader не заменяется CPU simulation.

Точная команда ручного ragged smoke:

```bash
build/metal-gqa-v10/astc-kv-probe-gqa-v10 \
  results/kernel_v2/fixtures_page/t391/manifest.json \
  metal/attention_gqa_v10.metal 1 1 13 - 20261102
```

Runner root:

```bash
PYTHON=/home/researcher/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$PYTHON" scripts/run_gqa_v10.py --out results/research_v4_20261029/gqa_main
```

Выше приведен путь завершенной серии. Для повторного запуска output directory должен быть новым. Во время серии не пересобирать binary и не менять shader/host/config/runner. Metadata подготовки, hashes и статическая проверка frozen inputs находятся в `build/metal-gqa-v10/preparation.json`.

## Результат завершенной серии

Read-only проверка `results/research_v4_20261029/gqa_main` подтвердила 304 attention gates и 38 exact decode/gather gates в 19 processes. Отдельный smoke содержит еще 80 attention gates и 10 exact gates в пяти processes. Наибольший relative L2 среди всех attention variants main равен 9.3074e-7. Все 2416 ABBA/BAAB blocks main содержат по четыре samples с ожидаемым порядком и одинаковым iterations внутри block. Это 2400 timing blocks и 16 blocks из четырех ragged correctness calls.

Для T8192 диапазоны пяти session medians B/A:

| Contrast | Диапазон |
|---|---:|
| ASTC GQA3 / прежний FP16 | 1.9801-2.1323 |
| ASTC GQA3 / FP16 GQA3 | 1.0570-1.1587 |
| ASTC GQA3 / прежний hoisted ASTC | 1.2612-1.3116 |
| FP16 GQA3 / прежний FP16 | 1.6936-1.9003 |

Primary timing criterion FAIL для обоих required contrasts во всех пяти sessions. Persistent memory criterion PASS: 3981312 / 6291456 = 0.6328125, то есть 36.71875% меньше FP16. Это прежняя экономия compact representation, а не новое уменьшение памяти от GQA3.

В этой реализации совместная загрузка K/V ухудшила время как ASTC, так и FP16. Близость ASTC к медленному FP16 GQA3 не считается успехом. Данные не устанавливают причину замедления и не опровергают GQA reuse вообще. Source-level сокращение повторных loads не является измерением physical traffic. Полные raw samples, включая широкие timing tails, сохранены. Итог эксперимента отрицательный; порог и controls не меняются.
