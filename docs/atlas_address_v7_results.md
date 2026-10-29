> Дополнение после отдельной ABBA-проверки: критерий не менее 5% выигрыша во всех пяти сеансах не пройден. Точность подтверждена. Результаты первой серии ниже сохранены; сопоставление обеих серий приведено в [итоговом отчете](research_v3_report.md).

# V7 address experiment: independent raw-result review

Проверены результаты исходного randomized ten-variant run в [address_v7](../results/research_v3_20261029/address_v7/summary.json). Зарегистрированный primary criterion выполнен: recurrent/v6 при T8192 имеет median within-round ratio <=0.95 во всех пяти сеансах. Hoisted дает похожий выигрыш. Дополнительного выигрыша recurrence относительно hoisted эти данные не показывают. Большой разброс отдельных раундов ограничивает выводы; надежный небольшой штраф recurrence не установлен.

Этот документ фиксирует только первоначальный run. Root готовит отдельное ABBA/BAAB confirmation из-за широких timing tails. Его результаты здесь не представлены. Исходные результаты, код и пороги не изменялись; новых GPU-команд при review не выполнялось.

## Что проверено

Источник - [registration.json](../results/research_v3_20261029/address_v7/registration.json), 15 полных raw runs, отдельный ragged run и все соответствующие `.execution.json`/`.stderr`. Условия: Apple M5 Pro, macOS 26.6.2 build25G83, 48 GiB unified memory; пять сеансов на T384/tile32, T2048/tile64, T8192/tile256. Каждый полный run содержит 60 раундов, 20 iterations на variant и 10 warmup calls. В каждом round присутствуют все десять variants ровно один раз, порядок 0..9 не повторяется.

PASS:

- 16/16 процессов завершились exit0, stderr пустой.
- Все 160 attention gates прошли с конечными значениями. Максимальный relative L2 среди всех десяти paths во всех полных runs - 9.307367557825085e-7 при пороге 1e-4.
- Во всех 16 runs compact normalized decode побитно совпал с original decode: одинаковые SHA256, max_abs=0. Все прежние value/decode gates тоже true.
- Ragged T391/tile13 прошел. У v6, hoisted и recurrent одинаковые error statistics: relative L2=3.9463430788178146e-7, max_abs=3.5762786865234375e-7.
- Все GPU durations положительны и конечны. В полных runs есть 600 samples на процесс; нет пропущенных или дублированных variant/round pairs.
- Hashes before и after совпали. Повторное чтение текущих frozen source, binary, config, driver и fixture files также совпало с registration.

У трех compact paths одинаковые error statistics на каждой форме. Это подтверждает прохождение численного gate, но не является отдельным доказательством побитного равенства attention outputs: raw JSON сохраняет error summaries, а не полные outputs. Побитное доказательство относится к normalized texture decode.

## Primary и secondary сравнения

Ratio вычислен как `gpu_batch_ms(numerator)/gpu_batch_ms(denominator)` для одного и того же round. Число iterations одинаково, поэтому деление на 20 сокращается. Это median отношений, а не отношение двух median latency.

T8192/tile256, 60 paired rounds на сеанс:

| Session | Recurrent/v6 | Hoisted/v6 | Recurrent/hoisted | Recurrent быстрее hoisted, rounds |
|---:|---:|---:|---:|---:|
| 0 | 0.907455 | 0.905916 | 1.001262 | 29/60 |
| 1 | 0.906190 | 0.904078 | 1.002502 | 29/60 |
| 2 | 0.906950 | 0.904331 | 1.006902 | 20/60 |
| 3 | 0.905302 | 0.901717 | 1.004198 | 24/60 |
| 4 | 0.906817 | 0.902774 | 1.003875 | 27/60 |

Primary recurrent/v6 дает уменьшение времени примерно на 9.25%-9.47% по зарегистрированной метрике. Secondary hoisted/v6 дает примерно 9.41%-9.83%. Recurrent/hoisted имеет median немного выше 1 во всех пяти сеансах, но только 129/300 отдельных pairs показывают recurrent быстрее. Это не свидетельство преимущества recurrence; величину порядка десятых долей процента не следует объявлять надежным slowdown по этим данным.

Secondary результаты других форм, диапазон session median ratios:

| T | Recurrent/v6 | Hoisted/v6 | Recurrent/hoisted |
|---:|---:|---:|---:|
| 384 | 0.958592-0.964081 | 0.952677-0.956022 | 1.003716-1.008655 |
| 2048 | 0.941076-0.945039 | 0.937706-0.941819 | 1.001883-1.004373 |
| 8192 | 0.905302-0.907455 | 0.901717-0.905916 | 1.001262-1.006902 |

Primary был заранее задан только для T8192 recurrent/v6. Остальные сравнения описательные и не подменяют его.

Для масштаба ниже приведена median пяти session medians в us на один amortized вызов. Это отдельная описательная агрегация; она не используется для primary ratio.

| T | V6 compact | Hoisted | Recurrent | Original-layout ASTC factored | Own FP16 |
|---:|---:|---:|---:|---:|---:|
| 384 | 8.217 | 7.842 | 7.892 | 7.712 | 5.733 |
| 2048 | 17.035 | 16.014 | 16.058 | 15.551 | 11.737 |
| 8192 | 43.825 | 39.659 | 39.758 | 38.179 | 25.336 |

При T8192 hoisted остается медленнее original-layout ASTC: paired median ratio 1.037782-1.042414. Относительно own FP16 ratio равен 1.561228-1.568295. Локальное улучшение compact addressing пока не дает выигрыша над этими контролями. Здесь нет сопоставления batched GPU time со standalone synchronous host latency llama.cpp.

## Широкие tails и границы вывода

В исходных raw samples есть большие колебания, данные не фильтровались. У recurrent/hoisted при T8192 session p10 лежит в диапазоне 0.348-0.581, session p90 - 1.553-3.163. Даже primary recurrent/v6 имеет session p10 0.366-0.539 и p90 1.339-2.905. Primary threshold <=0.95 проходит в 39, 40, 35, 35 и 42 rounds из 60, соответственно.

Отдельные абсолютные session medians тоже меняются: например, recurrent T8192 в session2 равен 46.740 us, а v6 в session3 равен 47.407 us, хотя median пяти session medians существенно ниже. Сравнения происходят в одном round, но не одновременно: это последовательные command buffers с randomized order. Pairing не удаляет весь кратковременный шум или nonlinear drift.

PMSet snapshots показывают AC power. Thermal output сообщает, что warning level не записан; это не измерение температуры и не доказательство отсутствия thermal pressure. По этим данным причина широких tails не установлена. Запланированный отдельный ABBA/BAAB run должен проверить устойчивость временного эффекта без удаления исходных samples.

## Интерпретация адресования и памяти

Mode1 сохраняет compact texture и логическую последовательность texture reads, но совместно меняет несколько вещей: выносит channel div/rem, head base и специализирует source width66 как 11 блоков. Поэтому допустим вывод, что этот набор изменений помогает в данном workload. Нельзя приписать весь эффект только одной div instruction или утверждать, что compiler прежде не делал hoisting.

Mode2 дополнительно заменяет token div/rem в loop на recurrent state. Он не превосходит mode1 по наблюдаемым session medians. Это полезный отрицательный результат для гипотезы дополнительного ускорения от recurrence; он не доказывает, какая instruction или зависимость ограничивает скорость.

Во всех 16 runs все pipelines, включая merge/decode, имеют одинаковые публичные свойства: `thread_execution_width=32`, `max_total_threads_per_threadgroup=1024`, `static_threadgroup_memory_length=0`. Эти значения не измеряют registers, occupancy, spills или hardware counters. Утверждение "register bottleneck" ими не поддерживается.

Compact representation между modes0/1/2 не меняется. При T8192 V texture занимает 737280 bytes, persistent K+V+metadata - 3981312 bytes; original-layout ASTC - 4816896 bytes, FP16 K+V - 6291456 bytes. Compact persistent sum на 36.71875% меньше FP16 sum. Это тот же allocation выигрыш layout v6, а не новый эффект v7. Все diagnostic resources одновременно живут в процессе, поэтому эти суммы не являются process peak или доказательством exclusive physical residency.

T2048 и T8192 - repeated-block traffic fixtures. Нет новых model-quality данных, encoding/append/update, full inference measurement или доказательства общего выигрыша на иных shapes/devices. На данном этапе обоснована проверка более простой hoisted формы; recurrence не следует продвигать как отдельно выигравшую оптимизацию.

## Точные команды и воспроизводимость review

Команды GPU уже были выполнены основным агентом. Например, session0 T8192, пути относительно корня репозитория:

```bash
build/metal-atlas-v7/astc-kv-probe-atlas-v7 \
  results/kernel_v2/fixtures_page/t8192/manifest.json \
  metal/attention_atlas_v7.metal 60 20 256 - 20269221
```

Ragged run:

```bash
build/metal-atlas-v7/astc-kv-probe-atlas-v7 \
  results/kernel_v2/fixtures_page/t391/manifest.json \
  metal/attention_atlas_v7.metal 1 1 13 - 20261029
```

Все исходные absolute argv сохранены в `.execution.json`. Entry point sweep - `scripts/run_address_sweep.py`; конфигурация - `configs/address_v7_20261029.json`. Повторный запуск требует нового output directory. В ходе этого review выполнялись только чтение JSON, hashes и расчеты statistics. Следующая read-only команда воспроизводит ключевое recurrent/hoisted сравнение и проверяет provenance:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, json, statistics as st
b = Path('results/research_v3_20261029/address_v7')
reg = json.loads((b/'registration.json').read_text())
summary = json.loads((b/'summary.json').read_text())
assert reg['hashes_before'] == summary['hashes_after']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
           for p, h in reg['hashes_before'].items())
for session in range(5):
    d = json.loads((b/f's{session}_t8192_tile256.json').read_text())
    names = ['astc_compact_factored_v', 'astc_compact_hoisted_v',
             'astc_compact_recurrent_v']
    g = {n: {s['round']: s['gpu_batch_ms'] for s in d['raw_samples']
             if s['variant'] == n} for n in names}
    assert all(len(v) == 60 for v in g.values())
    r = [g[names[2]][i]/g[names[1]][i] for i in range(60)]
    print(session, st.median(r), sum(x < 1 for x in r))
PY
```

Read-only raw audit: PASS. GPU correctness в сохраненных результатах: PASS. Registered primary criterion: positive. Независимый дополнительный выигрыш recurrence, register cause и ABBA confirmation: не установлены этим run.
