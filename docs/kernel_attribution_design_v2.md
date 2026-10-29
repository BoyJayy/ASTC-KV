# Metal v2: отделение загрузки от аффинного восстановления

Статус подготовки: реализовано; C++ host собран, shell syntax проверен. MSL-компиляция и GPU-выполнение v2 на этом этапе **не выполнялись**: GPU-серию запускает основной агент после измерения чувствительности v1 к split. Этот документ не содержит новых timing-результатов.

## Вопрос и контролируемые причины

Предыдущие три ветки одновременно меняли источник данных и наличие аффинного восстановления. v2 добавляет нормализованный FP16-буфер, чтобы сравнить текстуру и буфер при одинаковых метаданных, исходных числах и арифметике восстановления. Дополнительная обычная RGBA16Float-текстура проверяет влияние упаковки четырёх скаляров в один текстурный read.

| Вариант | Хранилище ASTC-стороны fixture | Восстановление внутри attention |
|---|---|---|
| `astc_texture` | Исходная ASTC texture | `affine4` |
| `uncompressed_texture` | Аппаратно декодированные нормализованные значения, округлённые в FP16, ordinary texture исходного layout | `affine4` |
| `fp16_buffer` | Уже восстановленные значения, затем округлённые в FP16 | Нет |
| `normalized_fp16_buffer` | Те же нормализованные FP16-скаляры, что в ordinary texture, линейный буфер `[Hkv,T,D]` | Тот же `affine4` |
| `uncompressed_rgba_texture` | Те же нормализованные FP16-скаляры, ordinary RGBA16Float texture, width=`D/4` | Тот же `affine4` |

Стороны fixture, уже хранящиеся в FP16, во всех вариантах используют один исходный FP16-буфер без нормализации. Q - один и тот же неизменяемый FP32-буфер для всех вариантов. Kernel, GQA, softmax, разбиение, scratch и merge сохранены. Оптимизации split, occupancy, кооперативного GQA, кодировщика или runtime не добавлены.

Пятый вариант **не кодирует RGBA ASTC**. Он проверяет ordinary RGBA16Float на тех же числах, поэтому не смешивает изменение упаковки с новой потерей кодека. Для входного `scalar` layout исходная ordinary texture читает четыре R16Float текселя, packed control - один RGBA16Float. Меняются также layout, padding и выделение texture; разницу нельзя приписать только числу shader-инструкций без дополнительного профилирования. При входном `rgba` layout это не сравнение четырёх scalar reads с одним vector read; JSON явно отмечает `scalar_vs_rgba_attribution_applicable=false`.

## Формат и отдельность версии

Новые исходники - `metal/attention_v2.metal`, `metal/probe_v2.mm`, `metal/build_v2.sh`. v1 не изменяется. Manifest остаётся `astc-kv-probe-v1`, output schema становится `astc-kv-metal-probe-result-v2` (ошибка - `astc-kv-metal-probe-failure-v2`). Params ABI остаётся 60 байт. Host поддерживает прежние D: положительное кратное 4, не больше 128; D64 и D128 входят в этот диапазон, но поддержка формы в коде не означает её аппаратную проверку.

MSL function constant 0 теперь `uint storage_path`: 0 = texture + affine, 1 = restored FP16 buffer, 2 = normalized FP16 buffer + affine. Host задаёт `MTLDataTypeUInt`. ASTC и ordinary textures используют один texture pipeline. Пятый control получает отдельные значения `components=4`, `width=D/4`; размеры attention и индексы metadata не меняются. Чтение texture остаётся по целочисленным координатам, level 0, без фильтрации.

Во время параллельного v1 опыта основной агент обнаружил failure path: неверная высота ASTC fixture отвергалась с открытым blit encoder, после чего Metal validation аварийно завершала процесс при его освобождении (`Command encoder released without endEncoding`). В v2 блок загрузки закрывает encoder при C++ exception и не commit-ит частичную загрузку. Требование `height=ceil(T/page_tokens)*page_tokens` не ослаблено. Исходный отрицательный запуск сохраняется основным агентом; исправление failure path требует отдельной аппаратной проверки на том же неверном manifest.

## Обязательные проверки до timing

1. Каждая из пяти attention-веток сравнивается с отдельным CPU reference с FP64-накоплением на её фактических значениях. Требуется конечность и relative L2 <= 1e-4.
2. Для каждой K/V стороны восстановленные ordinary texture и RGBA texture значения должны **точно** совпасть со значениями normalized FP16 buffer: `max_abs == 0`, обе стороны конечны. Это проверяет заявленный контроль одинаковых чисел независимо от attention.
3. Сохранён v1 gate software/hardware ASTC decode: для сжатой стороны требуется offline reference и `abs_delta <= scale/1024 + FP32_roundoff`; шкала конечна и положительна. Это ранее установленный инженерный smoke bound, не доказательство точного соответствия ASTC decoder или model-quality tolerance.
4. Для исходной FP16 стороны с offline values требуется точное совпадение. При провале JSON содержит диагностику и пустые `raw_samples`, процесс завершается ошибкой, timing не публикуется.

Нормализованные буферы и textures для контроля подготавливаются до timing. Исходный readback и все CPU references остаются диагностикой, не входят в измеряемые dispatch split+merge. Дополнительные ресурсы живут одновременно: process peak этого стенда не является memory peak производственного кэша. Новые выделения отдельно отражены как `normalized_buffer_control_allocated_bytes` и `ordinary_rgba_control_allocated_bytes`.

## CLI и измерение

```bash
bash metal/build_v2.sh
build/metal-v2/astc-kv-probe-v2 \
  results/metal_fixtures/story_l17_v_token6/manifest.json \
  metal/attention_v2.metal 30 10 256 - 20261028 \
  > NEW_OUTPUT.json 2> NEW_OUTPUT.stderr
```

Позиционные аргументы:

```text
MANIFEST METALLIB_OR_METAL_SOURCE [rounds=30] [iterations=10]
[tile_tokens=256] [readback_directory|-] [seed=20261028]
```

`--environment` сохранён. Необязательный readback directory должен существовать; `-` пропускает сохранение. `ASTC_KV_METAL_V2_BUILD_DIR` переопределяет отдельный build-каталог. Без offline Metal compiler host собирается через CLT и использует runtime compilation MSL3.0 без fast math; `shader_path.txt` указывает выбранный input. Успешная host-сборка не обозначает успешную MSL-компиляцию.

Для каждой ветки выполняются 10 warmup command buffers по одной итерации. Порядок всех пяти вариантов перемешивается в каждом warmup и измеряемом round через `std::mt19937(seed)` + `std::shuffle`; реальный порядок сохраняется в `order_within_round`. Каждый измеряемый round содержит все пять вариантов ровно по одному разу. Сохраняются batch GPU/wall durations, iterations, round, seed, warmup и cache-state. Между timed вариантами ждём завершения command buffer, как в v1; это диагностический предел измерения, не производственный dispatch schedule.

Первые три ветки v2 нужно измерить заново в том же сеансе: перенос старых v1 timings на v2 не предполагается. Для парных сравнений использовать round как единицу сопоставления внутри сеанса, дополнительно повторить независимые сеансы и проверить дрейф. При малом T384 кэш повторно читается и может помещаться в GPU cache; такой опыт не подтверждает большую пропускную способность памяти или ускорение генерации.

## Различающие результаты, зафиксированные до запуска v2

- `normalized_fp16_buffer` против `fp16_buffer`: показывает добавочную стоимость metadata/адресации/affine в текущем ядре; значения restored-half control могут дополнительно отличаться округлением, поэтому reference раздельный.
- `uncompressed_texture` против `normalized_fp16_buffer`: при exact-value gate отделяет ordinary texture path/layout от буферного доступа с тем же affine.
- `uncompressed_rgba_texture` против scalar `uncompressed_texture`: проверяет, помогает ли vector texture layout при тех же значениях и арифметике.
- `astc_texture` против ordinary texture controls: характеризует сжатый texture path, но сохраняет небольшое отличие дополнительного FP16 округления в ordinary controls.
- Если normalized buffer почти так же медленен, как scalar texture, объяснение исключительно texture load недостаточно. Если packed ordinary texture заметно быстрее scalar, это основание исследовать layout, но не доказательство пригодности RGBA ASTC по качеству.

Критерий полезности: различие должно воспроизводиться на парных раундах и не исчезать при смене порядка/сеанса; только после этого выбирается следующая локальная оптимизация. Без существенного устойчивого различия эти controls дают отрицательный результат для выбранного объяснения, а не повод заявлять ускорение.

## Проверка подготовки

Выполнены:

```bash
bash -n metal/build_v2.sh
bash metal/build_v2.sh
```

Обе команды PASS, host собран без предупреждений. Офлайн MSL-компилятор отсутствует; runtime MSL compilation, пять correctness gates и GPU timings - **ещё не выполнены**. GPU-прогон координирует основной агент, чтобы не смешать его с текущей серией v1.
