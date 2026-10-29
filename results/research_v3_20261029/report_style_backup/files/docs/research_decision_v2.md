# Независимое исследовательское решение v2

2026-10-28. Область: завершённые H1a/holdout, GPU v1–v4, common-Q comparison с llama.cpp и pool gate v5. Это read-only second opinion по сохранённым данным; автор этого документа новых измерений и внешнего поиска не выполнял. Common-Q baseline завершён, precision gates пройдены. Pool gate завершён в девяти process runs: его заранее заданный исход **negative**, независимая проверка сохранённых результатов **PASS**. История конкретизации протокола до первого pool measurement приведена ниже.

## Решение по текущей ветке

**Не заменять identity на transpose. NO-GO для дальнейшей разработки дорогостоящего GPU encoder/динамического cache под текущий scalar6-token/identity/D64 runtime путь.** Transpose не прошёл подтверждающий PPL gate; увеличение working set не позволило текущему предсжатому ASTC attention обогнать собственный FP16. На основной точке N64 ASTC медленнее примерно на 32% во всех трёх sessions, ещё без стоимости encoding/append. Это ограничение проверенного варианта, а не опровержение ASTC в целом, сторонних реализаций, других layouts, shapes или платформ. Дополнительный поиск гипотез и вариантов этой серией не предлагается.

В [H1a](block_geometry_v2.md) transpose уменьшил aggregate snapshot attention relative L2 на 2.9193% при прежних bits, но выбор сделан среди восьми D4 на старых данных. В [новой подтверждающей серии](geometry_holdout_v2.md) фиксированный transpose/identity PPL = **1.0011360376642466**, то есть заранее заданный критерий ≤1.0 не пройден. Три из четырёх новых документов хуже; корректность проверок PASS. Это отклонение текущего кандидата, а не статистическое доказательство вреда всех транспонирований. Одновременно поменялись данные и метрика, поэтому отдельно приписать неудачу domain shift или snapshot surrogate нельзя.

Identity ASTC V-only на новой серии имеет PPL/FP16 1.0132791 и 6 370 560 logical bytes; asymmetric affine U4 K/V — 1.0028735 и 4 020 480 bytes. В этой небольшой серии U4 лучше по этим двум измеренным осям. **Native latency U4 из CPU-результата не следует.** V-only ASTC сохраняет FP16 K; его нельзя описывать byte count прежнего U4-K/ASTC-V combo.

## Что действительно показали GPU-контроли

Источник — summaries [v1 split](../results/kernel_v2/split_v1_page/summary.json), [attribution](../results/kernel_v2/attribution/summary.json), [factored](../results/kernel_v2/factored/summary.json), [wave](../results/kernel_v2/wave/summary.json). Числа ниже — лучший tile по median session medians внутри завершённого v4 sweep; это подбор на измеренных данных, не независимая speed confirmation.

| T | v4 ASTC factored, µs | v4 own FP16, µs | ASTC/FP16 |
|---:|---:|---:|---:|
| 384 | 7.9073 | 5.8552 | ≈1.35 |
| 2048 | 15.6521 | 10.9167 | ≈1.43 |
| 8192 | 38.3698 | 25.3927 | ≈1.51 |

Attribution ослабляет простое объяснение «виноваты четыре texture reads»: при T384 normalized FP16 buffer с affine имеет 11.8104 µs против 11.7688 µs ASTC, тогда как готовый FP16 buffer — 8.9562 µs. Обычная RGBA texture даёт 11.6417 µs, а не устранение разрыва. При T8192 normalized buffer также медленнее обычного FP16. Это указывает на существенную роль affine/scheduling/register работы; **точная доля каждой причины и аппаратный bottleneck не установлены**.

Факторизация V и wave scheduling полезны инженерно, но ускорили и контрольные buffer paths. Текущий ASTC не обогнал собственный FP16. T2048/T8192 составлены повторением старых блоков, все серии многократно читают один cache: это traffic/shape controls, а не long-context model quality или доказанный DRAM-bandwidth режим. Own-reference correctness проверяет исполнение представленных значений; она не равна сохранению исходного model quality.

`codec_time_ms`, model latency и production peak в probe не измерены. CPU astcenc, загрузка/обновление страниц и динамический append не входят в приведённые microseconds. Прибавление этих расходов не создаёт выигрыша автоматически. Даже положительный следующий precompressed результат будет лишь основанием исследовать их бюджет.

## Завершённый официальный baseline на общем Q

[Common-Q summary](../results/kernel_v2/llama_comparison_common_q/summary.json) и [derived summary](../results/kernel_v2/derived_llama_summary.json) содержат девять завершённых llama.cpp GPU processes. Общий Q заранее одинаково округлён FP16→FP32, hardware-reconstructed K/V base совпадает по F16 hashes; Q4 повторно квантует этот base. Исходные thresholds не ослаблялись: max own-reference relative L2 F16 ≈2.02e−7, Q4 ≈5.84e−4, оба проходят свои frozen gates. Первоначальная серия с отличавшейся семантикой Q остаётся отдельной, её FAIL не переименован в PASS.

| T | ASTC host median, µs | llama Q4 host median, µs | llama F16 host median, µs |
|---:|---:|---:|---:|
| 384 | 156.0625 | 160.2505 | 155.8330 |
| 2048 | 164.0410 | 160.3330 | 155.9165 |
| 8192 | 188.0420 | 175.7710 | 167.8335 |

Здесь сравнивается одинаковый участок host submission→GPU completion, без timed output readback; framework/dispatch/synchronization overhead включён и доминирует. ASTC/Q4 session-median ranges перекрываются при T384/T2048. При T8192 они не перекрываются: ASTC 187.0835–192.6250 µs, Q4 169.4585–184.1255 µs. Сильного преимущества ASTC не установлено. Эти host времена нельзя делить на batched GPU-only времена из предыдущей таблицы или трактовать как различие чистой стоимости GPU kernels. Общий вход устраняет одну неоднозначность correctness; опыт не доказывает сопоставимость model quality двух codecs.

## Граница новизны

Локальные [prior-art](prior_art.md) и [независимый обзор](independent_research_20261028.md) уже фиксируют: Apple AFM [2507.13575v3](https://arxiv.org/html/2507.13575v3) применяет ASTC к весам; [KIVI](https://arxiv.org/html/2402.02750v2) описывает K/channel, V/token и FP residual tail; [Compute ASTC](https://niepp.github.io/2021/12/18/Compute-ASTC.html) показывает runtime GPU encoding; [Open-TQ-Metal](https://arxiv.org/abs/2604.16957v1) относится к прямому compressed attention на Apple Silicon. Эти первичные ссылки взяты из ранее проверенного обзора и сейчас заново не проверялись; числа сторонних работ здесь не воспроизведены.

Перестановки block layout, split attention/online softmax, SIMD scheduling и перенос `sum(p*(s*u+o))` в `sum((p*s)*u)+sum(p*o)` сами по себе не устанавливают новизну. Последняя формула — дистрибутивность для scalar per-token ranges. Известный принцип может быть частью полезной реализации, но это не новый математический механизм.

Потенциально отличимым вкладом остаётся **измеренно полезный активный ASTC KV на полном runtime пути** либо переносимый ASTC-specific механизм ошибки/выбора представления. Ни один из этих сильных результатов текущая серия ещё не установила. Отсутствие найденного точного аналога не доказывает мировой приоритет.

## Единственный дополнительный опыт: завершённый working-set gate

**Гипотеза:** часть текущего проигрыша v4 обусловлена режимом повторного чтения малого cache; при разных физических cache resources уменьшение трафика/рабочего набора ASTC компенсирует его вычислительную цену. Это проверка конкретного v4/identity/D64 пути, а не новый поиск представления.

**История конкретизации.** Первое предложение автора этого документа предусматривало N={1,16,64,128}, cap 2 GiB, latency ratio≤0.90 на N128 и отдельный memory criterion≤0.80. Оно **не исполнялось**. Координатор до любого pool measurement выбрал меньший первый gate и записал [configs/kernel_pool_v5.json](../configs/kernel_pool_v5.json): N={1,8,64}, cap 1 GiB и три исхода по latency. Причина уменьшения — ограниченный первый тест одной границы с примерно 400 MB FP16 working set; требование 20% ускорения соответствует принятой координатором исходной скоростной цели. Это отдельная конкретизация до результатов, а не утверждение, что исходное предложение всегда было тем же протоколом.

**Фактически замороженный и исполненный опыт:**

- Тот же precompressed T8192 fixture, Hq9/Hkv3/D64, identity orientation и неизменный v4 shader. Timing только `astc_factored_v` против собственного `fp16_buffer`, **tile256 у обоих**. Нет нового tile/layout/quality sweep.
- N={1,8,64}: отдельные resident allocations K-half, V-half, V-ASTC и metadata **между slots**; один полный обход пула в каждом timed batch. K buffer конкретного slot общий двум сравниваемым путям — это явно учтено в JSON, поэтому полной независимости pools между variants нет. Данные повторяют известный fixture. Максимум pool allocation — **1 GiB**. N64 FP16 payload составляет 384 MiB ≈402.7 MB до allocator rounding; фактические `allocatedSize` записываются отдельно.
- Три последовательные sessions, **30 rounds на N**, seed20261029; число вызовов в batch равно N. Основная метрика — GPU duration / N; сохранить raw samples, session medians, host time и pool memory. Основная точка — **N64**; N1/N8 диагностические, они не заменяют её после результата.
- До timing обязательны прежние own-reference gates всех семи базовых paths и выбранные pool-slot/wraparound checks. Failed control запрещает интерпретацию latency; tolerances не расширяются. Нет eviction command и аппаратного подтверждения miss rate: это large-working-set proxy, не измерение DRAM bandwidth.

**Положительный критерий:** на N64 ASTC/FP16 median GPU latency **≤0.80 в каждой из трёх sessions**. PASS разрешит исследовать бюджет отсутствующего активного runtime; он не докажет whole-model ускорение, стоимость CPU/GPU encoding, append или преимущество над llama.cpp.

**Отрицательный критерий:** на N64 отношение **>1.05 в каждой из трёх sessions**. Тогда NO-GO для разработки дорогого GPU encoder/append runtime **именно под этот текущий scalar6-token/identity/D64 путь**: даже предсжатый контроль с увеличенным рабочим набором остаётся медленнее собственного FP16. Это не опровержение всех ASTC layouts, моделей и аппаратных платформ.

**Иначе inconclusive** по заранее заданному gate: в частности, небольшой выигрыш или смешанные результаты нельзя переименовать ни в PASS, ни в NO-GO. Ошибка корректности, нарушение cap или incomplete run — технически невалидный опыт. Отдельный относительный memory threshold из первого предложения в фактический protocol не перенесён; добавлять его после измерений нельзя.

### Фактический исход

[Pool summary](../results/kernel_v2/pool_v5/summary.json) содержит все девять запланированных процессов. На основной точке N64:

| Session | ASTC GPU median/call, µs | FP16 GPU median/call, µs | ASTC/FP16 |
|---:|---:|---:|---:|
| 0 | 44.85221348 | 33.92773442 | 1.3219925893 |
| 1 | 44.87988281 | 34.01660160 | 1.3193523366 |
| 2 | 44.91992195 | 34.06673181 | 1.3185861853 |

Все три отношения выше 1.05, поэтому frozen gate даёт **negative**, не inconclusive. Результат не зависит от выбора лучшего N после измерения. N1/N8 также не показывают превосходства ASTC, но не используются для замены основной точки.

При N64 shader-read working sets по `allocatedSize`: ASTC 308 281 344 bytes, FP16 402 653 184 bytes; весь pool с общим K и обоими вариантами V — 509 607 936 bytes, ниже cap 1 GiB. Уменьшение учитываемого working set на 23.4375% не превратилось в выигрыш latency. Это описательная память, не дополнительный ретроспективный gate и не доказательство miss rate/DRAM bandwidth. Общие K, query/scratch, одинаковые данные и отсутствие hardware counters остаются ограничениями интерпретации.

Первичная регистрация runner не включала все input files. После запуска отдельный [verify_pool_results.py](../scripts/verify_pool_results.py) проверил фактически записанные runtime hashes без повторения benchmark: Q и K/V/ranges/ASTC payload одинаковы во всех девяти процессах и совпадают с исходным fixture; FP16 V дополнительно сопоставлен с отдельным ранее сохранённым hardware readback. [Независимая проверка](../results/kernel_v2/pool_v5/independent_validation.json) — **PASS**, проверены 219 slot hash sets, every-slot/wraparound coverage, schedule и raw medians. Максимальная pool own-reference relative L2 — 9.3074e−7. Эта последующая проверка воспроизводимости не выдаётся за первоначальную регистрацию входных файлов и не меняет thresholds или measurements.

Этот один gate не обнаружил измеримого запаса для отсутствующих encoding/cache-maintenance расходов в текущем пути. Совместно с отрицательным transpose holdout и отсутствием сильного преимущества над завершённым common-Q baseline он обосновывает ограниченный NO-GO выше. Более широкая полезность ASTC и upstream-работ этим экспериментом не опровергнуты; новая гипотеза потребовала бы отдельного обоснования и регистрации, а не продления текущего sweep после отрицательного исхода.

## Последующее наблюдение координатора после замороженного решения

Negative pool gate относится к прежнему 3-slice/66-wide atlas и сохранён. После него отдельные allocation/v6 опыты выявили проверенный trade-off: компактный атлас хранит те же ASTC blocks, даёт bit-exact hardware decode, на T8192 сокращает V allocation на53.125% и persistentKV на17.35% к старомуASTC, но замедляет attention на14.37–14.58%. Это новое техническое объяснение возможной экономии памяти, не опровержение прежних результатов и не основание объявить runtime speedup. Данные и ограничения в [основном отчёте](research_v2_report.md), раздел5.1. Этот абзац добавлен интегратором после независимого решения.
