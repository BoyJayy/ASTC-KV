# ASTC-KV: ближайшие работы и границы возможного вклада

Проверено: 2026-10-28. Статус: целевой обзор первичных источников.

Дополнение после независимой перепроверки: [новые источники, актуальные ревизии
baseline и уточнённая граница новизны](independent_research_20261028.md).
Карточки ниже сохраняют содержание исходного обзора v0.1.

Прямого аналога активного KV-кэша с прямым чтением ASTC в attention в проверенных источниках не найдено; мировое первенство не установлено.

## 1. Что уже известно и что предстоит доказать

Использование аппаратного ASTC для нейросетевых тензоров уже показано Apple; runtime-кодирование ASTC уже существует. Сжатие KV видеокодеками, асимметрия K/V, взвешивание ошибки, ортогональные преобразования, in-kernel dequantization, остаточный FP-хвост и paged storage тоже не являются новыми идеями сами по себе. Ссылка в каждой строке ведёт к первичному источнику; код анализировался статически, аппаратные результаты авторов здесь не воспроизводились.

Возможный вклад - совместное проектирование активного ASTC KV, ограниченного кодировщика и прямого attention, дающее измеримый выигрыш относительно сильного low-bit Metal пути. Более сильная гипотеза - дешёвая характеристика конкретного ASTC-представления, которая предсказывает его пользу на отложенных запросах, слоях или моделях.

## 2. Представление и путь данных

| Работа | Что / когда кодируется | Где хранятся данные | Декодирование | Прямое compressed attention | Полная декодированная копия |
|---|---|---|---|---|---|
| [Apple Intelligence Foundation Language Models][apple_2025] | Веса серверной модели: ASTC HDR-ch 6x6; KV отдельно квантуется до 8 бит. После обучения; используются quality-recovery adapters. | ASTC-блоки весов и FP16-минимум каждого блока. | Аппаратный ASTC-декодер Apple GPU; минимум добавляется внутри tensor operations. | ASTC KV-attention не описан. | Отдельная полная реконструкция весов не требуется по описанию. |
| [LLM.265: Video Codecs are Secretly Tensor Codecs][llm265] | Веса, KV, активации и градиенты; HEVC/H.264 после 8-битной квантизации. Веса офлайн; активации/KV/градиенты при исполнении. | Видео-битстримы; runtime учитывает восстановленные тензоры. | NVDEC -> тензоры; опубликованный pipeline возвращает torch.stack(decoded). | Нет продемонстрированного attention, читающего битстрим непосредственно. | Декодированные тензоры материализуются; размер одновременно живущей копии всего KV не установлен. |
| [Efficient Remote KV Cache Reuse with GPU-native Video Codec][kvcodec] | KV-фрагменты для удалённого повторного использования; codec-friendly layout. Заранее, до remote reuse. | Удалённое хранилище -> host bitstream buffer -> GPU frames. | NVDEC и frame-wise restoration в paged KV vLLM. | Нет: inference получает восстановленный KV. | Рабочий paged KV восстановлен; временный decoder buffer сокращён покадровой обработкой. |
| [Compute ASTC][compute_astc] | LDR-текстуры, ограниченное подмножество ASTC 4x4/6x6. Runtime compute shader. | Валидные блоки ASTC, затем сжатая текстура. | Обычный аппаратный texture decoder. | Нет attention. | Не нужна для графического texture sampling. |
| [Arm ASTC Encoder][arm_astcenc] | Изображения/текстуры, LDR/HDR, 1-4 компоненты. Офлайн CPU reference/tool; пригоден для начальной оценки качества. | 128-битные ASTC-блоки и контейнер. | Программная reference-декомпрессия или аппаратная поддержка API/GPU. | Нет KV runtime. | CPU validation может материализовать изображение; GPU sampling не требует полной копии. |
| [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache][kivi] | Активный KV: K per-channel, V per-token; остаточный FP16-буфер. Онлайн группами. | Packed low-bit GPU tensors, scale/zero-point и residual cache. | CUDA: dequantization объединена с matrix multiplication; Triton quantization. | Да для QK/PV matmul; это не обязательно одно fused softmax-attention ядро. | Для основного low-bit пути полная FP16-копия не требуется. |
| [KVQuant][kvquant] | KV; K до RoPE; per-channel/per-token, nonuniform codebooks, sparse outliers. Калибровка кодовых книг офлайн; квантизация KV онлайн. | Packed dense+sparse KV на GPU. | Собственные CUDA matvec kernels с dequantization/RoPE. | Да, fused low-bit matvec-путь. | Не требуется в соответствующих kernels. |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs][quarot] | Веса, активации, KV после ортогональных/Hadamard-преобразований. Преобразование весов офлайн; activation/KV quantization онлайн. | Low-bit GPU tensors. | CUDA low-bit operators, dequantization по мере потребления. | Представлен low-bit KV decode path; детали fusion зависят от operator. | Полная persistent FP16-копия KV не является требованием метода. |
| [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate][turboquant] | Векторы/KV; rotation, scalar codebooks, MSE/product variants. Онлайн; без калибровки на конкретном наборе данных. | Коды векторов, нормы и необходимые вспомогательные данные. | Программные vector estimators/quantization operators. | Работа с кодами для logits описана; готовый сопоставимый fused Metal KV backend не проверен. | Для оценивания dot products не нужна; полная интеграция требует отдельной проверки реализации. |
| [PagedAttention / vLLM][pagedattention] | Исходный вклад - управление размещением KV, не lossy codec. Динамическое выделение/сопоставление страниц при inference. | Несмежные GPU blocks, block tables, shared prefixes. | Attention адресует физические blocks через таблицы. | Квантизация не является центральным условием исходной работы. | Копия contiguous KV не нужна. |
| [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving][flashinfer] | Attention engine для разных KV layouts; в статье также FP8/FP16 mixed precision. Prefill/append/decode в runtime; собственно encoder зависит от формата. | Paged/ragged/block-sparse, composable shared-prefix storage. | CUDA/CUTLASS attention, преобразование tiles внутри kernel. | Да для поддерживаемых low-precision путей; не ASTC. | Полная FP16-копия cache не требуется для on-the-fly пути. |
| [MLX / MLX-LM current public implementation][mlx_lm] | QuantizedKVCache: packed KV и scale/bias; FP cache тоже поддержан. update_and_fetch онлайн; расширение capacity порциями 256. | MLX GPU arrays, uint32 payload и metadata. | Quantized matmul -> scores/softmax -> quantized matmul. FP branch использует mx.fast SDPA. | QK и PV получают packed KV; весь attention не является единым fused kernel в этом Python-пути. | Python-путь не разворачивает полный KV; scores материализуются. |
| [llama.cpp Metal quantized FlashAttention][llamacpp_metal] | KV Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, FP16 и другие разрешённые варианты. Runtime KV update; конкретные сочетания необходимо проверить запуском. | GGML block-quantized buffers. | Metal FlashAttention templates вызывают dequantize_q4_0 внутри вычисления. | Да: kernel_flash_attn_ext и vec; есть специализация D=64. | Полная FP16-копия не требуется для этих kernel templates. |
| [Open-TQ-Metal][open_tq_metal] | Активный K/V: affine INT4, FP16 scale/zero на группу 32, то есть 5 эффективных бит/скаляр до остальных расходов. Онлайн; после FP16 prefill и при добавлении каждого decode token; в Llama code append через concatenate. | Packed Metal/MLX arrays и FP16 metadata в unified memory. | Metal register unpack внутри SDPA, online softmax, split-K partial/reduce; Llama kernel специализируется на D=128. | Да: реальный опубликованный packed-INT4 attention kernel; не ASTC. | В fused decode полная FP16 KV-копия не нужна; prefill FP16 и partial/reduce buffers остаются. |

## 3. Проверка на системе и сопоставление с ASTC-KV

| Работа | Аппаратура | Что измерено | Код / воспроизводимость | Отличие предполагаемого ASTC-KV |
|---|---|---|---|---|
| [Apple Intelligence Foundation Language Models][apple_2025] | Apple Private Cloud Compute; точная конфигурация ASTC-microbenchmark не раскрыта. | Качество моделей; отдельного воспроизводимого сравнения ASTC KV decode latency нет. | Открытого кода этого ASTC backend в источниках не найдено. | Активный append-only KV, онлайн-кодирование и attention вместо фиксированных весов. |
| [LLM.265: Video Codecs are Secretly Tensor Codecs][llm265] | NVIDIA; в опытах обучения RTX3090; отдельные предложения аппаратуры моделируются. | Качество/размеры, обучение; section 6.1: примерно 1.1/1.3 GB/s NVENC/NVDEC, признаны ограничением. | Открытый codec pipeline и quality-evaluation scripts; полная active-KV интеграция не подтверждена ревизией. | Произвольный доступ через текстурный тракт и отсутствие внешнего восстановления KV. |
| [Efficient Remote KV Cache Reuse with GPU-native Video Codec][kvcodec] | NVIDIA A100, H20, L20; несколько GPU, сети 1-40 Gbps. | TTFT, TPOT, качество и память remote reuse; до 3.51x TTFT в исследованных сценариях. | Ссылка на авторский открытый репозиторий в проверенной статье/поиске не найдена. | Сжатое resident-представление для каждого последующего attention, включая динамическое добавление. |
| [Compute ASTC][compute_astc] | В статье iPhone6s+/iPhone12; опубликованный repo использует D3D11. | PSNR текстур и GPU-время кодирования; не K/V, не LLM. | Авторский код доступен, исходный backend D3D compute. | K/V-специфичный отбор режимов, attention-error objective и окупаемость обновлений. |
| [Arm ASTC Encoder][arm_astcenc] | CPU SIMD; аппаратный ASTC проверяется отдельно. | Reference codec; собственных K/V-измерений нет. | Открытый авторский codec и документация, Apache-2.0. | Reference oracle, а не научный baseline готовой active-KV системы. |
| [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache][kivi] | NVIDIA A100 80GB в системных опытах. | Качество, peak memory, throughput при увеличении допустимого batch. | Открыт; CUDA/Triton, не подтверждённый Metal baseline. | Fixed-function ASTC, layout и стоимость encoding/upload. |
| [KVQuant][kvquant] | NVIDIA A100; расширение памяти до нескольких GPU. | PPL, memory capacity и matrix-vector kernel timings; отдельные границы контекста. | Открыт; Metal-путь не установлен. | Взвешенная ошибка и исключения уже известны; нужен ASTC-специфичный алгоритм. |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs][quarot] | NVIDIA GPU; не Apple Metal. | PPL, zero-shot accuracy, latency и memory. | Открытый авторский репозиторий. | Само применение ортогонального преобразования не является новым вкладом. |
| [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate][turboquant] | Статья: A100; блог: H100 для logits timing. | LongBench/NIH и искажения; блог: до 8x logits vs FP32 keys, не полный decode vs FP16. | В проверенных авторских источниках официальный Metal-код не найден; независимые реализации существуют. | ASTC-ограничения, аппаратный decoder, online обновление и все накладные расходы. |
| [PagedAttention / vLLM][pagedattention] | NVIDIA GPU serving systems. | Throughput, latency, memory fragmentation/sharing. | Открыт. | Страницы ASTC и FP-tail - применение известных идей управления памятью. |
| [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving][flashinfer] | NVIDIA Turing-Hopper в статье. | Kernel и end-to-end serving latency, распределения workload. | Открытый авторский engine. | Слияние частичных attention и разные layouts известны; новый предмет - аппаратный ASTC путь. |
| [MLX / MLX-LM current public implementation][mlx_lm] | Apple Silicon/Metal; на нашей Linux-среде не выполнен. | Здесь проверен исходный код, не latency. | Открытый авторский runtime; ревизии зафиксированы. | Практический совместимый baseline; сравнивать также с fused Metal low-bit implementation. |
| [llama.cpp Metal quantized FlashAttention][llamacpp_metal] | Apple Silicon/Metal; фактический dispatch на данном Mac ещё не измерен. | Здесь проверено существование и устройство kernels; сравнительные замеры запланированы. | Открыт; CLI: --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0. | Сильный Metal baseline для конкуренции с fixed-function decode. |
| [Open-TQ-Metal][open_tq_metal] | Автор сообщает M1 Max, 64 GB, 32 GPU cores, macOS 15; здесь статический аудит, без аппаратного повторения. | Опубликованы synthetic kernel tests до 128K и короткая model generation. Объявленные 48x и full-context capacity требуют отдельной проверки; см. аудит ниже. | Два открытых авторских репозитория. В Llama benchmark слабый unfused reference, в model prefill отсутствует causal mask; baseline нельзя принять без исправлений. | Аппаратный ASTC decoder, специализированное представление и экономия полного пути; само fused compressed Metal attention уже не кандидат на новизну. |

## 4. Ошибки, которые нельзя переносить в наше сравнение

- **Apple Intelligence Foundation Language Models:** 128/36 = 3.56 - payload; с одним FP16-минимумом (128+16)/36 = 4.00 бит/вес до остальных расходов.
- **LLM.265: Video Codecs are Secretly Tensor Codecs:** Не переносить модельные будущие ускорения кодеков в измерения существующего GPU.
- **Efficient Remote KV Cache Reuse with GPU-native Video Codec:** Свойства временной/пространственной избыточности KV и влияние раскладки уже исследованы.
- **Compute ASTC:** Перенос существующего кодировщика на Metal сам по себе не новизна.
- **Arm ASTC Encoder:** Биты/texel не равны битам/исходный KV-скаляр при разном распределении компонент.
- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache:** Batch-throughput из статьи не считать ускорением batch=1 decode на Mac.
- **KVQuant:** Не переносить заявленную скорость отдельных matvec на полный decode.
- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs:** Rotation может улучшать scalar quantization, но ухудшать пространственную гладкость для ASTC.
- **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate:** section 4.3 v1 называет (32*3+96*2)/128 значением 2.5, хотя это 2.25; перепроверять фактическую бухгалтерию битов.
- **PagedAttention / vLLM:** Нельзя заявлять новизной только pages, append-only layout или prefix sharing.
- **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving:** CUDA результаты не служат сравнением скорости с Apple GPU.
- **MLX / MLX-LM current public implementation:** Наличие quantized_matmul не доказывает, что этот путь оптимален для полного decode.
- **llama.cpp Metal quantized FlashAttention:** 4-битный payload Q4_0 имеет FP16 scale на 32 значения: минимум 4.5 бит/скаляр до padding.

- **Open-TQ-Metal:** 48x относится к несопоставимому unfused kernel reference, не к оптимизированному FP16 или q4 runtime; 128K end-to-end и качество не подтверждены просмотренными тестами.

### Бухгалтерия ASTC

Для блока из `Bx * By` texel с `c` **различными полезными** KV-скалярами в каждом texel payload составляет `128/(Bx*By*c)` бит/скаляр. Если один скаляр продублирован в RGB, `c=1`, а не 3. Добавляются метаданные, padding, таблицы, хвост и одновременно живущие копии. Наличие RGBA не означает четыре независимых кодировщика: оценивать такой packing нужно на настоящих данных.

Apple-подобный scalar 6x6 с одним FP16 minimum имеет 4.00 бит/скаляр до прочих расходов. LDR с FP16 scale + FP16 offset имеет `(128+32)/36 = 4.44` бит/скаляр до padding. Вариант только V при FP16 K соответственно имеет среднюю ставку `(16+bV)/2`, например 10 бит/скаляр K/V при `bV=4`. Это расчёты размера представления, не измерение выделенной памяти Metal.

### Наблюдения о коде и версии

- В pinned MLX-LM `base.py` low-bit путь состоит из двух `mx.quantized_matmul` и отдельного softmax, FP-путь вызывает `mx.fast.scaled_dot_product_attention`. Это реальная baseline-реализация, но её нельзя заранее объявить сильнейшей.
- В pinned llama.cpp присутствуют `fa_q4_0.metal` и `fa_vec_q4_0.metal`, в том числе D=64. Основной template распаковывает блоки внутри attention. Перед сравнением скорости надо проверить фактический GPU-dispatch и отсутствие fallback на выбранной модели.
- В pinned LLM.265 `pytorch_nvenc/pipeline/nvenc.py` восстанавливаются обычные тензоры. Примеры quality evaluation записывают восстановленные веса обратно в модель. Это не подтверждает наличие опубликованного memory-efficient active-KV runtime.

### Open-TQ-Metal: аудит опубликованного baseline

Это важный прямой аналог **активного сжатого KV attention на Metal**. Его отличие от ASTC-KV - программная распаковка обычного INT4. Наличие такого кода сужает допустимую формулировку новизны: одного Metal kernel, который избегает полной декомпрессии, недостаточно. Для численных и временных сравнений кандидат полезен после исправления harness и причинности. Ниже результаты статического аудита Llama repository, а не утверждение о выполнении этих исходников на нашем устройстве.

1. **Заявленные 48x не относятся к сильному FP16 baseline.** В `sdpa_int4_bench.cpp` reference повторяет K/V по GQA-головам и использует отдельные matmul, softmax и matmul. `mx::fast::scaled_dot_product_attention` не используется. Распакованные K/V создаются до таймера, поэтому надпись "dequantize then attend" не описывает реально включённые стадии. Fused path прогрет и повторён пять раз, reference замерен один раз. Нужны одинаковый прогрев, одинаковые входы после FP16-метаданных и сильные FP16/low-bit operators. [Код benchmark](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/tests/sdpa_int4_bench.cpp).

2. **Опубликованный model prefill теряет причинность.** Для `S>1` `llama_model.cpp` передаёт в fast SDPA пустой `mask_mode` и не передаёт mask array; до этого идут только projections, reshape и RoPE. FP16 fallback устроен аналогично. Перед переносом модели необходимо исправить causal mask и проверить непрерывный и chunked prefill; совпадение двух одинаково неверных путей не проверяет модель. [Model code](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/engine/llama_model.cpp). [Официальный MLX fast.cpp: causal mode задаётся явно](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/fast.cpp).

3. **Проверенные тесты не показывают полную 128K генерацию модели.** `llama_e2e.cpp` запускает 5 prompt и 200 decode tokens, затем экстраполирует KV bytes к 131072. `context_scaling.cpp` проверяет prompts до 512. `kv_memory_test.cpp` суммирует sizes массивов, живущих по одному слою, а для длинных случаев использует 10 слоёв, поэтому не измеряет peak всего 80-layer KV. В этом тесте нельзя приравнивать сумму логических размеров к resident process memory. [E2E](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/tests/llama_e2e.cpp), [memory test](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/tests/kv_memory_test.cpp).

4. **Граница реализации и метаданные существенны.** Llama fused kernel специализируется на `D=128`; `D!=128` в model code возвращается к FP16 KV. Два FP16 параметра на 32 INT4 scalar дают `4+32/32=5` эффективных бит/скаляр, то есть 3.2x от FP16. Append через `concatenate` и split-K partial/reduce buffers также требуют учёта в latency и peak. Это не готовый D=64 baseline для первого SmolLM2 без адаптации. [Metal kernel](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/engine/sdpa_int4.metal), [model code](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/engine/llama_model.cpp).

5. **Опубликованные числа качества и скорости недостаточны для наших критериев.** В `context_scaling.cpp` таймер охватывает четыре decode forward calls, но формула использует число пять. Synthetic attention test использует broad mean-absolute-error threshold; наблюдения top-1/coherent text не заменяют held-out NLL, retrieval и самостоятельную генерацию. Отдельная теоретическая оговорка: Eq.6 статьи не общая формула angular score error. Для `dot(q,k)=0` поворот `k` к `q` даёт изменение, пропорциональное `sin(delta_theta)`, а не `1-cos(delta_theta)`. Изолированная scale-константа не учитывает нормы Q/K, а `rho^L` не доказанная модель переноса ошибки через residual transformer. [Context timing](https://github.com/mutable-state-inc/turboquant-llama3.170B/blob/e3dd4ca41d2e028dea3569dcc36b9055915f9968/tests/context_scaling.cpp), [статья](https://arxiv.org/pdf/2604.16957v1).

Решение: сохранить Open-TQ-Metal в обзоре и при наличии Mac воспроизвести исправленный kernel benchmark. Основной runtime baseline выбирать по фактически корректному и быстрому исполнению; текущие MLX-LM и llama.cpp остаются обязательными кандидатами. Не переносить 48x, projected capacity или "identical top-1" в заключение ASTC-KV.

## 5. Кандидаты на вклад и опровергающие проверки

| Кандидат | Уже известная часть | Что могло бы быть новым | Что опровергнет полезность |
|---|---|---|---|
| Scalar ASTC для V, затем K отдельно | ASTC weights, asymmetric KV | Layout и нормализация под ASTC при фактическом бюджете ~4-5 бит | Ни один режим не проходит quality gate относительно q4 после метаданных |
| Предиктор пригодности страницы/головы | KV statistics, spatial/temporal layouts, error weighting | ASTC-specific interpolation/partition residual предсказывает ошибку attention на held-out данных | Перестановки не меняют результат либо предиктор не переносится на новые запросы |
| Ограниченный GPU encoder | Compute ASTC | Сокращение поиска по статистике KV с проверяемой потерей качества и latency | Потери качества или обслуживание cache уничтожают выигрыш |
| Прямое texture attention | Apple fixed-function decoding, fused quantized attention | Исполнение активного append-only KV без полной декомпрессии и лучший Pareto на Mac | ASTC уступает fused q4/q8 при равном качестве и рабочем объёме |
| Политика момента сжатия | Residual caches, paging, amortization | Решение без знания будущего на основе измеренной стоимости и локальных признаков | Выигрыш исчезает на коротком continuation или новой нагрузке |

Начинать следует с численной проверки scalar-представлений, ограниченного набора осей группировки и permutations с обратным восстановлением порядка. Перестановка должна сохранять сами числа, а внимание после восстановления должно использовать исходную позиционную семантику. Одновременно можно проверять texture sampling заранее сжатых блоков: это благоприятный контроль конкретной реализации, не строгая верхняя граница любого будущего ASTC-пути.

## 6. Зафиксированные версии и источники

### Apple Intelligence Foundation Language Models

arXiv: 2507.13575v3, 2025-08-27; Apple article updated 2025-07-17

- [https://arxiv.org/html/2507.13575v3](https://arxiv.org/html/2507.13575v3)
- [https://machinelearning.apple.com/research/apple-foundation-models-2025-updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)

### LLM.265: Video Codecs are Secretly Tensor Codecs

MICRO 2025 camera-ready; code 262e588b11937801a2b3bd0ce760ba05abc4da32

- [https://www.yongjiwu.me/assets/pdf/micro25-llm265.pdf](https://www.yongjiwu.me/assets/pdf/micro25-llm265.pdf)
- [https://github.com/Entropy-xcy/llm.265/tree/262e588b11937801a2b3bd0ce760ba05abc4da32](https://github.com/Entropy-xcy/llm.265/tree/262e588b11937801a2b3bd0ce760ba05abc4da32)

### Efficient Remote KV Cache Reuse with GPU-native Video Codec

arXiv: 2602.09725v3, 2026-05-12; SIGCOMM 2026

- [https://arxiv.org/html/2602.09725v3](https://arxiv.org/html/2602.09725v3)

### Compute ASTC

Author post 2021-12-18; code 7cd3af99a01a3889347cc7c8d62699e247b638ca

- [https://niepp.github.io/2021/12/18/Compute-ASTC.html](https://niepp.github.io/2021/12/18/Compute-ASTC.html)
- [https://github.com/niepp/astc_encoder/tree/7cd3af99a01a3889347cc7c8d62699e247b638ca](https://github.com/niepp/astc_encoder/tree/7cd3af99a01a3889347cc7c8d62699e247b638ca)

### Arm ASTC Encoder

main d0f91386350222fcbf6da533c0f7a3f9b5c6b783, 2026-10-21; use separately pinned release for experiments

- [https://github.com/ARM-software/astc-encoder](https://github.com/ARM-software/astc-encoder)
- [https://github.com/ARM-software/astc-encoder/blob/main/Docs/Encoding.md](https://github.com/ARM-software/astc-encoder/blob/main/Docs/Encoding.md)
- [https://github.com/ARM-software/astc-encoder/blob/main/Docs/FormatOverview.md](https://github.com/ARM-software/astc-encoder/blob/main/Docs/FormatOverview.md)

### KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

arXiv: 2402.02750v2, 2024-07-25; code 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6

- [https://arxiv.org/html/2402.02750v2](https://arxiv.org/html/2402.02750v2)
- [https://github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)

### KVQuant

arXiv: 2401.18079v6, 2025-05-28; code 57a238357f0ffe50084670fcd5781c9848f80ea2

- [https://arxiv.org/pdf/2401.18079v6](https://arxiv.org/pdf/2401.18079v6)
- [https://github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant)

### QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

arXiv: 2404.00456v2, 2024-10-29; code 5008669b08c1f11f9b64d52d16fddd47ca754c5a

- [https://arxiv.org/pdf/2404.00456v2](https://arxiv.org/pdf/2404.00456v2)
- [https://github.com/spcl/QuaRot](https://github.com/spcl/QuaRot)

### TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

arXiv: 2504.19874v1, 2025-04-28; author blog 2026-03-24

- [https://arxiv.org/html/2504.19874v1](https://arxiv.org/html/2504.19874v1)
- [https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### PagedAttention / vLLM

arXiv: 2309.06180v1, 2023-09-12

- [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

### FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

arXiv: 2501.01005v2, 2025-04-21

- [https://arxiv.org/html/2501.01005v2](https://arxiv.org/html/2501.01005v2)
- [https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)

### MLX / MLX-LM current public implementation

MLX-LM 1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7; MLX 7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a; 2026-10-28

- [https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/base.py](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/base.py)
- [https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/cache.py](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/cache.py)
- [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

### llama.cpp Metal quantized FlashAttention

ac8e1694d837b11ed0aab944a6a2eee34b9c4d80, 2026-10-28

- [https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_q4_0.metal](https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_q4_0.metal)
- [https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_vec_q4_0.metal](https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_vec_q4_0.metal)
- [https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_vec_common.metal](https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_vec_common.metal)

### Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon

arXiv: 2604.16957v1, submitted 2026-04-18; PDF masthead 2026-04-21; Llama code e3dd4ca41d2e028dea3569dcc36b9055915f9968, 2026-04-14; Gemma code 0f09466b7fde772a4876bf7bee3ccdeb34313304, 2026-04-13

- [https://arxiv.org/abs/2604.16957v1](https://arxiv.org/abs/2604.16957v1)
- [https://arxiv.org/pdf/2604.16957v1](https://arxiv.org/pdf/2604.16957v1)
- [https://github.com/mutable-state-inc/turboquant-llama3.170B/tree/e3dd4ca41d2e028dea3569dcc36b9055915f9968](https://github.com/mutable-state-inc/turboquant-llama3.170B/tree/e3dd4ca41d2e028dea3569dcc36b9055915f9968)
- [https://github.com/mutable-state-inc/gemma4metal/tree/0f09466b7fde772a4876bf7bee3ccdeb34313304](https://github.com/mutable-state-inc/gemma4metal/tree/0f09466b7fde772a4876bf7bee3ccdeb34313304)

## 7. Поиск точных аналогов

Дата поиска: 2026-10-28. Применялись два поисковых движка, затем чтение первоисточников. Запросы и исходы сохранены в `prior_art.json`.

- system1: `"ASTC" "KV cache"`; `"ASTC" "attention" compression cache`; `"texture compression" "key-value" cache`. Exact active ASTC KV attention match not found in retrieved primary sources.
- system2: `"ASTC" "KV" attention github`; `"ASTC" "cache" LLM Metal`; `"texture" "KV cache" compression Apple`. Apple weights and an MLX-LM ASTC weight feature request; no verified active KV implementation.
- system1: `site.research.google "TurboQuant" 2026`; `"KVCodec" github "Liang"`; `"ASTC" "KV-cache"`. Author TurboQuant March 2026 blog located; no verified exact ASTC KV analogue.

- Дополнительный system1-поиск root выявил Open-TQ-Metal (arXiv: 2604.16957); проверены статья и опубликованный код. Прямой active INT4 Metal путь найден; точного ASTC аналога это не устанавливает.

Ограничение: поиск не охватывает закрытые реализации, все патенты, недоступные preprints и неиндексированные репозитории. Неполнота выдачи не доказывает отсутствия работы. При обнаружении совпадения пересматривается формулировка вклада, а не замалчивается источник.

[apple_2025]: https://arxiv.org/html/2507.13575v3
[llm265]: https://www.yongjiwu.me/assets/pdf/micro25-llm265.pdf
[kvcodec]: https://arxiv.org/html/2602.09725v3
[compute_astc]: https://niepp.github.io/2021/12/18/Compute-ASTC.html
[arm_astcenc]: https://github.com/ARM-software/astc-encoder
[kivi]: https://arxiv.org/html/2402.02750v2
[kvquant]: https://arxiv.org/pdf/2401.18079v6
[quarot]: https://arxiv.org/pdf/2404.00456v2
[turboquant]: https://arxiv.org/html/2504.19874v1
[pagedattention]: https://arxiv.org/abs/2309.06180
[flashinfer]: https://arxiv.org/html/2501.01005v2
[mlx_lm]: https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/base.py
[llamacpp_metal]: https://github.com/ggml-org/llama.cpp/blob/ac8e1694d837b11ed0aab944a6a2eee34b9c4d80/ggml/src/ggml-metal/kernels/fa_q4_0.metal
[open_tq_metal]: https://arxiv.org/abs/2604.16957v1
