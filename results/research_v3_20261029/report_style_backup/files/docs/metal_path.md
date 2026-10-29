# Metal: проверка аппаратного пути ASTC-KV

Дата проверки документации: 2026-10-28.

**Историческая спецификация v0.1. Обновление:** аппаратные проверки на M5 Pro
выполнены в продолжении; см. [актуальный отчёт](revalidation_report_20261028.md)
и [подробности сборки](metal_revalidation_20261028.md). Ниже сохранено исходное
состояние и ограничения на момент CPU-пилота.

**Статус на момент исходного пилота: исходники реализованы; компиляция и выполнение на Apple GPU не выполнены.**
Рабочая среда подготовки — Linux x86_64, без Xcode, Metal SDK и Apple GPU.
Этот документ не содержит результатов измерения GPU и не подтверждает ускорение.

## Что подтверждено первичными источниками

| Вопрос | Проверенный ответ | Источник |
|---|---|---|
| Может ли compute читать ASTC? | Таблица Metal разрешает чтение/семплирование поддерживаемых форматов из graphics и compute; ASTC присутствует у Apple GPU. | [Metal Feature Set Tables, 21 May 2026, pp. 3, 11, 14](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) |
| Может ли compute писать произвольные тексели ASTC? | Для ASTC нет возможности `Write` в таблице. Путь обновления должен получать готовые ASTC-блоки отдельно. | Тот же источник |
| Можно ли перенести блоки из GPU-буфера в texture? | `MTLBlitCommandEncoder` предоставляет buffer→texture copy. Для сжатого формата stride задан между строками блоков; область должна соответствовать границам блоков, с исключением края texture. | [Apple buffer-to-texture copy](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy%28from%3Asourceoffset%3Asourcebytesperrow%3Asourcebytesperimage%3Asourcesize%3Ato%3Adestinationslice%3Adestinationlevel%3Adestinationorigin%3Aoptions%3A%29?language=objc) |
| Нужен ли CPU `replaceRegion`? | Нет для выбранного пути. Прототип загружает готовые блоки через staging buffer и blit в private texture. `replaceRegion` имеет отдельные правила синхронизации и ограничение для private storage. | [Apple replaceRegion](https://developer.apple.com/documentation/metal/mtltexture/replace%28region%3Amipmaplevel%3Awithbytes%3Abytesperrow%3A%29) |
| Как измеряется время GPU? | Разность `GPUEndTime-GPUStartTime` после завершения command buffer; временные метки до завершения могут быть нулевыми. | [Apple GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime) |
| Как узнать профильные счётчики? | Перечислить `MTLDevice.counterSets` и проверить поддерживаемую точку выборки. Наличие конкретного счётчика не предполагается. | [Apple counter availability](https://developer.apple.com/documentation/metal/confirming-which-counters-and-counter-sets-a-gpu-supports) |
| Как учитывать физические ресурсы? | Сохранять `MTLResource.allocatedSize`; отдельно считать логические payload/metadata и RSS всего процесса. | [Apple allocatedSize](https://developer.apple.com/documentation/metal/mtlresource/allocatedsize) |

Размер `.astc` файла, payload блоков, staging stride, выделенная texture и пиковая память процесса — разные показатели. Загрузка готовых блоков **не является ASTC-кодированием**.

## Реализованная конструкция

Прототип `metal/probe.mm` независимо выбирает для K и V ASTC texture либо исходный FP16 buffer. Для сжатой стороны он загружает по одному `.astc` на KV-голову в слои `texture2d_array`. Он использует `ASTC_*_LDR`, без sRGB и mipmaps. Поддержаны квадратные блоки 4×4, 6×6, 8×8.

В `attention.metal` одно SIMD-подмножество из 32 потоков вычисляет один контекстный фрагмент одной query-головы. Каждый активный поток обслуживает четыре канала. По токенам выполняется устойчивый online softmax. Второе ядро объединяет частичные нормировочные коэффициенты и взвешенные суммы. Интермедиат имеет размер O(Hq × splits × D), а не O(T × D) распакованного KV.

Каждое текстурное чтение использует integer-coordinate `read(..., slice, level=0)`. Нежелательной фильтрации между соседними значениями нет. После текстурного чтения применяется аффинное восстановление `offset + scale × normalized`. Метаданные реально лежат в FP16-буферах и читаются как `half`; умножение и накопление выполняются в FP32. Поддержаны независимые режимы нормализации K/V: scale/offset на канал страницы (`page_channel`) либо одна пара на весь вектор токена (`token`). Сторона с FP16 storage не имеет affine metadata и читается напрямую из буфера.

Область реализации намеренно ограничена:

- batch=1, один запрос к attention, все `tokens` являются допустимым причинным префиксом;
- Q/K уже прошли нужное позиционное преобразование; прототип не применяет RoPE повторно;
- GQA через `kv_head = query_head / (Hq/Hkv)`;
- `D` кратен четырём и не больше 128;
- заранее сжатые стороны K/V и/или полные FP16-стороны; без динамического кодирования и свежего хвоста;
- page_tokens кратно высоте ASTC-блока;
- одинаковый размер atlas K и V, если обе стороны сжаты;
- высота atlas ограничена максимальной высотой 2D texture конкретного GPU; длинные кэши за этим пределом потребуют отдельной адресации страниц;
- SIMD width 32 проверяется при создании pipeline.

Режимы K-only и V-only задаются смешанным storage. Динамические обновления, политика выбора форматов и оптимизированная интеграция с MLX остаются последующими этапами.

## Интерфейс данных: astc-kv-probe-v1

Примеры схемы: `metal/manifest.example.json` и `metal/manifest_v_only.example.json`. Это примеры полей, а не поставляемые наборы входных данных.

| Поле | Тип и смысл |
|---|---|
| `format` | Ровно `astc-kv-probe-v1` |
| `tokens` | Число реальных KV-токенов T; padding исключён из attention |
| `query_heads`, `kv_heads`, `head_dim` | Hq, Hkv, D; Hq делится на Hkv |
| `page_tokens` | P; число страниц `ceil(T/P)` |
| `layout` | `scalar` или `rgba` |
| `metadata_dtype` | Ровно `float16_le` |
| `k_storage`, `v_storage` | Независимо `astc` (по умолчанию) либо `fp16` |
| `k_normalization`, `v_normalization` | Для ASTC: `page_channel` (по умолчанию) либо `token`; для FP16 игнорируется |
| `query_file` | little-endian float32, C-order `[Hq,D]` |
| `k_files`, `v_files` | Для соответствующей ASTC-стороны: массивы из Hkv имён `.astc`, в порядке голов |
| `k_file`, `v_file` | Для соответствующей FP16-стороны: little-endian float16 `[Hkv,T,D]`; `*_files` и affine-файлы не нужны |
| `k_offset_file`, `k_scale_file` | Для ASTC K: little-endian float16; `page_channel` → `[Hkv,ceil(T/P),D]`, `token` → `[Hkv,T,1]` |
| `v_offset_file`, `v_scale_file` | Те же правила для ASTC V, с независимым `v_normalization` |
| `k_decoded_file`, `v_decoded_file` | Необязательные little-endian float32 `[Hkv,T,D]`, уже восстановленные офлайн; только для проверки CPU/GPU decoder agreement |

Все имена файлов разрешаются относительно директории manifest. Модель, ревизия, слой и другие сведения могут храниться в дополнительных полях: входной manifest целиком включается в результат.

`scalar`: texel `(x=c,y=t,slice=h)` содержит значение в R; при кодировании рекомендуется `rrr1`. `rgba`: texel `(x=floor(c/4),y=t,slice=h)`, компонент `c mod 4`. Для footprint Bx×By ширина файла равна `ceil((D/components)/Bx)*Bx`, высота — `ceil(T/P)*P`. Неиспользованные каналы/строки кодируются как padding и учитываются в payload. В `page_channel` номер страницы метаданных равен `floor(t/P)`. В `token` metadata индекс равен `h*T+t`; метаданные padded-строк отсутствуют, поскольку они не читаются attention. Значения метаданных округляются до FP16 **до** нормализации/кодирования.

Arm документирует scalar mapping `rrr1`, поскольку это позволяет использовать luminance endpoint modes. RGBA распределяет четыре скаляра по компонентам, но независимые компоненты могут хуже соответствовать модели endpoints/weights ASTC. Это обоснование эксперимента, не обещание его результата. [Arm Effective ASTC Encoding](https://github.com/ARM-software/astc-encoder/blob/main/Docs/Encoding.md)

## Разделение ошибки кодека и ошибки ядра

Перед измерением прототип выполняет отдельное аппаратное чтение ASTC в диагностический CPU-буфер. Полученные нормализованные числа восстанавливаются с теми же FP16-метаданными и служат входом CPU attention с FP64-накоплением. Затем сравнивается результат прямого ASTC attention. Критерий реализации: конечные значения и относительная L2-ошибка не более 1e-4.

Этот readback находится **вне измеряемого пути attention**. Он оставляет полную диагностическую копию только в процессе проверочного стенда; прототип отдельно сообщает этот факт в памяти. Он не нужен для непосредственного выполнения shader attention и не является частью предложенного активного кэша.

Контрольные варианты:

| Вариант результата | Хранилище | Контрольная цель |
|---|---|---|
| `astc_texture` | ASTC private textures + FP16 affine metadata для сжатых сторон; исходные FP16 buffers для остальных | Прямое чтение смешанного кэша и восстановление в attention |
| `uncompressed_texture` | Сжатые стороны заменены на R16Float для scalar либо RGBA16Float для rgba; FP16-стороны сохраняются | Выделить влияние сжатого текстурного пути при аналогичной адресации и нормализации |
| `fp16_buffer` | Сжатые стороны предварительно восстановлены и округлены в FP16; исходные FP16-стороны сохраняются | Сравнить buffer access в той же простой архитектуре ядра |

Каждый вариант сверяется с CPU attention на **своих фактических значениях**, поскольку FP16-буфер и несжатая texture могут внести дополнительное округление. Опциональные `*_decoded_file` отделяют различия программного и аппаратного ASTC-декодирования от ошибки attention.

Эти два контроля **не являются сильными оптимизированными baseline MLX/llama.cpp**. Даже выигрыш относительно них не доказывает улучшение inference.

## Запуск на Mac

Из корня проекта:

```bash
bash metal/build.sh
bash metal/probe_environment.sh > results/mac_environment.json
build/metal/astc-kv-probe path/to/manifest.json \
  build/metal/attention.metallib 20 10 256 > results/metal_probe.json
python metal/summarize.py results/metal_probe.json > results/metal_probe_summary.json
```

Нужно создать `results/`, если её ещё нет. Требуются Apple Silicon, macOS 13+ и выбранный Xcode с Metal compiler. Компиляция отключает fast math для начальной проверки корректности; выбранный `-std=metal3.0` не требует Metal 4.

Аргументы после `.metallib`: число раундов, повторений в command buffer, токенов в split. После каждого полного command buffer выполняется один wait; после каждого маленького kernel CPU-синхронизации нет. Внутри command encoder установлены buffer barriers между зависимыми dispatch. [Apple memory barriers](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier%28scope%3A%29)

Перед timing выполняется проверка корректности и пять прогревочных итераций каждого варианта. Порядок вариантов разворачивается через раунд. Сохраняются сырые GPU/wall batch times; отдельный скрипт строит медиану и bootstrap-интервал. Длительность каждой итерации вычисляется как длительность batch / iterations, поэтому это амортизированный kernel throughput, не наблюдаемая задержка токена модели.

Измерения многократно читают тот же кэш. Малые T могут целиком помещаться в аппаратный кэш; их нельзя переносить на длинный контекст. Необходимо повторить на допустимых моделью больших T и во взаимодействии с реальными модельными kernels. Полноценная оценка latency, decode, TTFT, prefill, энергорежима, thermal drift и динамической цены кодирования остаётся вне первого probe.

## Память и интерпретация результатов

JSON сообщает отдельно payload блоков, размер файлов, FP16-стороны кэша, реально использованные FP16-метаданные, выделенные texture/buffer ресурсы, staging и scratch attention. Логические bits/scalar включают все padding-строки/колонки и знаменатель из исходных реальных K+V скаляров. Для смешанного кэша числитель содержит FP16-данные несжатой стороны; дополнительно публикуются отдельные эффективные биты для K и V. Поэтому V-only ASTC не называется полностью низкобитным KV.

`control_params_bytes` отдельно сообщает 60 байт структуры параметров, не добавляя её к логическому payload кэша. Неактивные shader arguments получают безопасные dummy bindings; их фактические выделения отдельно учитываются в `inactive_binding_allocated_bytes`, но не выдаются за affine metadata. `fp16_decode_control_only_allocated_bytes` выделяет распакованные диагностические буферы сжатых сторон; исходные FP16-стороны учтены в `fp16_cache_allocated_bytes`.

`production_peak_bytes` остаётся null: процесс одновременно держит диагностические CPU-копии и GPU-контроли. `device_current_allocated_bytes_all_diagnostics` — снимок текущих выделений, а `process_max_rss_bytes_macos` — высокий уровень RSS всего проверочного процесса. Ни один из этих показателей нельзя выдать за пик будущего сжатого runtime. Сами выделения Metal могут иметь дополнительные внутренние издержки, не отражённые логическим payload.

## Следующий аппаратный эксперимент

1. На реальном Mac собрать код и сохранить environment; исправить обнаруженные SDK/компиляционные проблемы.
2. Сначала проверить маленький настоящий KV fixture: CPU/GPU decode agreement, GQA, неполная страница и неполный split.
3. Получить correctness pass для нескольких T, footprint и двух layout.
4. Измерить выбранное качество/представление на больших допустимых контекстах.
5. Добавить сильный FP16 и quantized-KV runtime baseline с теми же Q/K/V; только затем оценивать потенциал выигрыша конкретного пути.

До шага 3 аппаратная корректность обозначается «не подтверждена»; до шагов 4–5 нет основания утверждать ускорение.

## Выполненная доступная проверка исходников

В Linux выполнены `bash -n` для двух shell-скриптов, `py_compile` для скрипта сводных измерений, чтение обеих JSON-схем и проверка их базовых ограничений. Статически сопоставлены порядок всех 15 полей Params в C++ и MSL и размер 60 байт. `probe_environment.sh` корректно завершился с кодом 2 и статусом `metal_gpu_status=unavailable_in_current_environment`. Это проверка вспомогательных файлов, она не заменяет компиляцию Objective-C++/MSL и аппаратные correctness-тесты.
