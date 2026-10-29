> Обновление интегратора: основной GPU sweep (9 processes) и T391/tile13 smoke выполнены, exact normalized decode и attention gates PASS. T8192: V allocation -53.125%, GPU latency +14.37-14.58% к старому ASTC layout. Raw `results/kernel_v2/atlas/`, `atlas_ragged/`; [полный итог](research_v2_report.md). Ниже сохранён исходный дизайн подготовки.

# Compact ASTC block atlas: v6 execution probe

Статус подготовки: новый host собран существующим CLT без предупреждений; CPU inverse-address property check прошёл. GPU compilation, decode и attention агент подготовки не запускал. Их PASS можно установить только по аппаратным результатам основного агента.

Предшествующий allocation-only опыт сохранён в `results/kernel_v2/atlas_allocation.json`. На T8192 переход от 66x8256x3 к compact 192x8514x1 уменьшил заявленный Metal `allocatedSize` с 1572864 до 737280 bytes; на T2048 - с 393216 до 196608 bytes; на T384 остался 49152 bytes. Эти observations мотивируют проверку реального codec path, но сами не подтверждают decode или speedup. Отрицательный performance result исходного pool layout сохраняется.

## Изменение представления

Новые файлы: `metal/attention_atlas_v6.metal`, `metal/probe_atlas_v6.mm`, `metal/build_atlas_v6.sh`; build - `build/metal-atlas-v6`. Исходные v4 files и results не меняются. Поддержан только bounded experiment: D64, scalar ASTC 6x6 LDR, три KV heads, исходная width66, FP16 K и token-normalized ASTC V.

Host читает прежние `.astc` payloads в порядке head -> block row -> block column. Каждый 16-byte блок копируется без изменения в линейную последовательность с 32 блоками, или 512 bytes, на строку. Создаётся private ASTC texture шириной 192 texels, высотой `6*ceil(total_blocks/32)`, array length1. Если последняя строка неполная, оставшиеся места заполняются копией первого валидного блока. Логический tensor их не адресует. В трёх зарегистрированных fixtures дополнительные padding blocks не требуются.

CPU byte compare исходной последовательности и packed staging обязателен; SHA256 обоих payloads сохраняется в JSON. Blit upload завершается до decode, correctness и timings. Нового ASTC encoding нет. Bounds: texture dimensions <=16384, staging <=64 MiB, дополнительные блоки <=20% исходного количества.

Исходные `Params.width/height` не заменяются размерами compact texture: они по-прежнему задают width66 и padded height исходного atlas. Для каждого отдельного scalar channel `c`:

```text
block = (head*(source_height/6) + token/6)*(source_width/6) + c/6
x = (block % 32)*6 + c%6
y = (block / 32)*6 + token%6
slice = 0
```

Четыре компоненты `float4` переводятся независимо. Нельзя вычислить одну позицию и прибавить 0...3: float4 может пересекать 6-channel block boundary и конец строки компактного atlas. Используется padded source height, включая последнюю неполную страницу T8192.

## Shader и correctness

Shader - копия v4 с новым function constant2 `compact_atlas`. Host явно задаёт false во всех прежних attention pipelines; true получает только новая specialization с factored V. ABI `Params`, Q, K, metadata, split/merge, диапазоны и reference не меняются. Исходные `texture4`, `attention_merge` и `decode_texture` сохранены. В K-ветке compact addressing не применяется; K остаётся FP16.

Независимый scalar kernel `decode_compact_texture` разворачивает весь логический normalized V. Перед timing host требует finite values и **побитное совпадение FP32** с исходным hardware normalized V (`memcmp`), сохраняет SHA256 обоих массивов. Это более строгий gate, чем совпадение только attention output. Если он не пройден, timings не публикуются.

Все семь прежних attention paths, их own-reference gates, exact ordinary/RGBA/normalized comparisons и offline decode checks сохраняются. Восьмой path `astc_compact_factored_v` (index7) использует ту же ASTC CPU FP64 accumulation reference, что исходная ASTC texture; требование finite и relative L2 <=1e-4 остаётся прежним. Ни пороги, ни reference по результату не меняются. Старый invalid-height manifest должен по-прежнему отвергаться с завершением открытого blit encoder и exit1.

## Timings и память

По умолчанию 10 warmup calls на каждый из восьми вариантов; затем 30 rounds по 10 iterations. Порядок всех восьми вариантов перемешивается seeded shuffle в warmup и каждом измеряемом round. Граница GPU/wall batch сохраняется от v4. CPU repack, upload, independent decode, references и pipeline creation находятся вне attention timing. Это не runtime append/update или model latency.

JSON содержит source и compact размеры, blocks/capacity, padding bytes, payload SHA, preparation/upload time, actual compact texture allocation, staging size, а также отдельные persistent sums:

- compact K + compact V texture + V metadata;
- исходные K + original V texture + V metadata;
- FP16 K + FP16 V;
- compact persistent + общий query/scratch.

Прежние scratch/control/device allocation и process RSS поля сохраняются. Обе ASTC textures, все control resources и CPU references живут одновременно; per-path persistent sum не равна process peak или отдельной физической residency. Снижение allocation не гарантирует скорость: remap добавляет integer arithmetic и меняет locality.

## Проверки подготовки и запуск

```bash
bash -n metal/build_atlas_v6.sh
bash metal/build_atlas_v6.sh
```

Обе команды PASS, host build без предупреждений. Offline `metal` compiler отсутствует; shader будет собран runtime с Metal3/no-fast-math при разрешённом root GPU run. CPU inverse-address check проверил 2152128 logical coordinates для T1/96/97/384/391/2048/8192, включая float4 block/atlas-row seams и partial pages. `build/metal-atlas-v6/address_validation.json` сохраняет покрытие. Это CPU geometry evidence, не аппаратная проверка shader. Полный diff просмотрен; старые merge/decode kernels и value/decode gate blocks сравнены без изменений.

Короткая аппаратная проверка основным агентом:

```bash
build/metal-atlas-v6/astc-kv-probe-atlas-v6 \
  results/kernel_v2/fixtures_page/t384/manifest.json \
  metal/attention_atlas_v6.metal 1 1 32 - 20261029
```

Зарегистрированный sweep использует root-owned `configs/kernel_atlas_v6.json`: T384/tile32, T2048/tile64, T8192/tile256; три сеанса, 30 rounds, 10 iterations. Width32 выбран по завершённому allocation-only исследованию до compact attention measurements.

```bash
python3 scripts/run_kernel_sweep.py \
  --config configs/kernel_atlas_v6.json \
  --binary build/metal-atlas-v6/astc-kv-probe-atlas-v6 \
  --shader metal/attention_atlas_v6.metal \
  --fixtures results/kernel_v2/fixtures_page \
  --out results/kernel_v2/atlas_v6
```

Output directory должен быть новым. Driver сохраняет negative manifest outcome, raw results, stdout/stderr и hashes. Source/binary/config preparation hashes также сохранены в `build/metal-atlas-v6/preparation.json`. До завершения аппаратного run не заявляются codec equivalence, attention PASS или speedup. Любой slowdown и исходные отрицательные результаты сохраняются.
