> Обновление интегратора: GPU-серия выполнена, все runtime gates и post-run audit PASS; предзаданный performance outcome NEGATIVE (ASTC/FP16 N64=1.322/1.319/1.319). Raw `results/kernel_v2/pool_v5/`, подробности в [общем отчёте](research_v2_report.md). Ниже сохранён исходный дизайн подготовки.

# Resident pool probe v5

Статус подготовки: host собран существующим CLT без предупреждений. Shader остаётся ровно `metal/attention_v4.metal`. Новые GPU-команды агент подготовки не выполнял; аппаратная проверка и измерения переданы основному агенту. Наличие build не означает PASS GPU.

Цель - проверить, скрывал ли повторный обход одного маленького KV cache выигрыш сжатого представления. Этот опыт меняет адреса резидентных ресурсов при одинаковых значениях. Он не измеряет качество кодека, реальную загрузку модели, append/encoding или DRAM bandwidth. Большой pool сам по себе не доказывает определённую долю cache misses.

## Контракт

Новые файлы: `metal/probe_pool_v5.mm`, `metal/build_pool_v5.sh`, этот документ; build - `build/metal-pool-v5`. V4, baseline и прежние результаты не менялись. Shader SHA256 при подготовке: `a3f81b1aaa5c96ac75a9011f8ea767694be74ec6e91bc084df1fdf527850a5fa`.

```text
astc-kv-probe-pool-v5 MANIFEST METALLIB_OR_METAL_SOURCE
    [rounds=30] [iterations=pool_size] [tile_tokens=256]
    [readback_directory|-] [seed=20261028] [pool_size=1]
```

`pool_size` - целое 1-64; `iterations` обязан равняться `pool_size`. Допустимы только D64, FP16 K, ASTC V с token normalization. Остальные shape, metadata, finite/decode checks унаследованы от v4. Parent driver `scripts/run_pool_sweep.py` и регистрация `configs/kernel_pool_v5.json` задают T8192/tile256, pools 1/8/64, 30 раундов и три сеанса; эти файлы принадлежат основному агенту.

Каждый slot содержит отдельные `newBuffer` K half, V half, V offset/scale и отдельную private ASTC texture. K одного slot общая для двух веток; ASTC V и восстановленная FP16 V используют разные ресурсы. Pool не использует heaps или aliasing. Проверяются уникальные resource objects и отсутствие alias с исходными buffers. Все значения копий одинаковы, выделения разные.

До создания pool его размер оценивается по фактическому `allocatedSize` исходных однотипных ресурсов. Затем сумма фактических новых allocations проверяется после каждого slot; допустимо не более 1 GiB на весь pool, включая обе V representation и K/metadata. Исходные диагностические ресурсы учитываются отдельно и также остаются живы.

## Проверки до измерений

1. Все семь исходных v4 attention paths проверяются против собственных FP64 accumulation references, finite и relative L2 <= 1e-4. Exact ordinary/RGBA/normalized values и исходные decode gates сохраняются.
2. Каждая ASTC texture копируется full-slice texture-to-texture blit из исходной texture; общий command buffer завершается до дальнейших действий.
3. SHA256 каждого K/V/offset/scale buffer совпадает с исходным. Raw compressed texture bytes читаются blit-ом в CPU buffer, без decode: хешируются только payload block rows, padding исключён. Исходный texture payload сравнивается с конкатенацией payload исходных `.astc` файлов, а каждый slot - с ним. В JSON сохраняются хеши всех slot.
4. Обе измеряемые ветки выполняются по одному разу на **каждом** slot, output сравнивается с соответствующим исходным reference тем же численным gate. После полного обхода проверяется wrap к первому slot, затем результат последнего slot второго обхода. При N=1 первая и последняя позиции совпадают.

Если attention gate не пройден, печатается `correctness_failed` и пустые timings. Ошибка allocation/byte-integrity останавливает процесс с ненулевым exit до warmup/timing; stderr сохраняет причину. Значения slot одинаковы, поэтому attention error сам по себе не доказывает адресную уникальность: её проверяют создание отдельных ресурсов, resource identities и код cursor/binding.

## Измеряемая работа

Только paths 5 (`astc_factored_v`) и 2 (`fp16_buffer`). Для каждой ветки отдельный cursor; на каждой итерации выбирается `cursor % pool_size`. **Внутри** цикла перед dispatch повторно привязываются K, V half, V offset/scale и обе texture arguments. Число и порядок вызовов binding одинаковы; различается attention pipeline. Неактивные resources всё равно привязаны, но соответствующая shader specialization их не читает.

После correctness cursors сбрасываются. Warmup - один полный pool cycle на ветку. Каждый измеряемый batch - ещё один полный cycle, один compute encoder/command buffer, split+merge и те же buffer barriers, что v4. Порядок двух веток перемешивается с фиксированным seed в warmup и каждом из 30 раундов. Raw samples сохраняют variant, round, order, iterations, cursor before, first/last slot, GPU batch duration и synchronous wall batch duration.

GPU batch / N - амортизированное время вызова в batch. Wall batch / N тоже амортизировано и не сопоставимо с отдельным synchronous llama host-call. Ни копирование pool, ни хеширование, ни readback, ни reference не входят в timing. Отдельного принудительного eviction нет. Порядок веток и общий K pool могут влиять на cache state; три сеанса и исходные raw samples нужны для оценки устойчивости.

## Учёт памяти

`pool.slots` содержит actual `allocatedSize` K, V half, V ASTC, offset и scale на каждый slot. `memory` отдельно сохраняет:

- весь pool и суммы K, FP16 V, ASTC V, metadata;
- shader-read working set ASTC = K + ASTC V + metadata, FP16 = K + FP16 V;
- те же суммы с общими query/scratch;
- counts ресурсов pool и общего query/partial/stats/output;
- уникальную сумму исходных diagnostic resources и сумму pool + исходные ресурсы;
- transient texture-verification buffer, все прежние control allocations, `device.currentAllocatedSize` и process max RSS.

Shader-read working set не означает отдельную физическую residency: обе ветки и семь исходных diagnostic representations сосуществуют. Pipeline/driver allocations отражает общий device counter; CPU reference tensors - process RSS. Эти поля не являются production peak.

## Проверенные команды и передача запуска

```bash
bash -n metal/build_pool_v5.sh
bash metal/build_pool_v5.sh
```

Обе команды PASS. Build запускает существующий `xcrun clang++` с C++17, ARC, `-O3 -Wall -Wextra`, Foundation и Metal. Offline `metal` compiler отсутствует; `shader_path.txt` указывает на исходный v4 shader для runtime Metal3 compilation без fast math. Новых зависимостей нет.

Короткая статическая проверка: исходные `run`, references и attention/decode gate blocks совпадают с v4; полный diff проверен на ограничение двух timed variants, полный cycle, одинаковые bindings и память. Build/provenance и hashes сохранены в `build/metal-pool-v5/preparation.json`. GPU correctness, raw ASTC readback и pool performance пока **не проверены этим агентом**.

Точная одиночная команда для основного агента (вывод направлять в новый файл):

```bash
build/metal-pool-v5/astc-kv-probe-pool-v5 \
  results/kernel_v2/fixtures_page/t8192/manifest.json metal/attention_v4.metal \
  30 64 256 - 20261029 64
```

Полный зарегистрированный sweep, только основным агентом:

```bash
python3 scripts/run_pool_sweep.py --out results/kernel_v2/pool_v5
```

Driver создаёт output directory с `exist_ok=False`, сохраняет stdout/stderr каждого запуска и хеши до/после. Если каталог уже существует, выбрать новый; прежние результаты не перезаписывать. Откат подготовки прост: не запускать новый binary; прежние v4 executable/source остаются самостоятельными и не менялись.
