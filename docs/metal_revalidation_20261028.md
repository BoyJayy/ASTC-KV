# Независимая аппаратная проверка Metal, 28 октября 2026

Это дополнение к исходному v0.1, а не изменение истории его результатов. Первый реальный запуск подтвердил аппаратное чтение ASTC и корректность attention на одном готовом fixture. Полный активный KV-кэш, кодировщик GPU, обновления страниц, модельная генерация и преимущество над оптимизированным baseline этим запуском не проверены.

**Итог после интеграции:** выполнены оба исходных fixture, raw readback,
производные K-only/FP16 controls, три ожидаемых отказа и полный launcher.
Подробности и актуальные выводы - в [общем отчёте](revalidation_report_20261028.md).
Ниже ранние этапы оставлены с их фактическим статусом на момент записи.

## Окружение и воспроизводимость

Из `results/revalidation_20261028/metal/environment_before_build.json`:

- Apple M5 Pro, 20 GPU cores, 48 GB unified memory;
- macOS 26.6.2 (25G83), SDK 26.5, arm64;
- выбран `/Library/Developer/CommandLineTools`, полный Xcode и утилита `metal` отсутствуют;
- питание от сети, `lowpowermode=0` на момент снимка.

Исходная команда `bash metal/build.sh` завершилась кодом 2. Сообщение сохранено в `build_original.log`: выбранный набор инструментов не предоставляет Metal compiler. Это ограничение офлайн-сборки, а не отсутствие GPU.

Host был успешно собран без установки компонентов:

```bash
xcrun -sdk macosx clang++ -std=c++17 -O3 -fobjc-arc -mmacosx-version-min=13.0 \
  -Wall -Wextra metal/probe.mm -framework Foundation -framework Metal \
  -o results/revalidation_20261028/metal/astc-kv-probe-original
```

Первый запуск `--environment` внутри песочницы сообщил `Metal device is unavailable`. После разрешённого запуска вне песочницы обнаружено настоящее Metal-устройство; результат - `device_original_unsandboxed.json`. Эти два состояния не следует смешивать.

В `probe.mm` добавлен путь для входного `.metal`: официальный [`newLibraryWithSource:options:error:`](https://developer.apple.com/documentation/metal/mtldevice/makelibrary%28source%3Aoptions%3A%29?language=objc), MSL 3.0, `fastMathEnabled=NO`. Исходный путь загрузки `.metallib` сохранён. Компиляция исходного шейдера происходит до измеряемого attention, её время входит в `pipeline_creation_wall_ms`.

## Первый выполненный fixture

Точная команда успешного запуска:

```bash
results/revalidation_20261028/metal/astc-kv-probe-runtime \
  results/metal_fixtures/story_l17_v_token6/manifest.json \
  metal/attention.metal 1 1 256 \
  > results/revalidation_20261028/metal/runtime_original_v.json \
  2> results/revalidation_20261028/metal/runtime_original_v.stderr
```

Запуск выполнен вне песочницы после разрешения доступа к GPU. Fixture: story, слой 17, T384, Hq9/Hkv3, D64, FP16 K и заранее сжатый ASTC6/token V, весь V сжат без точного хвоста. В этом первом запуске ещё использовался исходный warmup - один пакет из пяти повторений на вариант; последующая правка заменяет его десятью отдельными command buffers.

| Проверка | Фактический результат |
|---|---:|
| ASTC attention против CPU FP64 на аппаратно декодированных значениях, relative L2 | 4.012374933252955e-7 |
| Та же проверка, max absolute error | 8.046627044677734e-7 |
| FP16 buffer control, relative L2 | 5.042098720514364e-7 |
| Ordinary texture control, relative L2 | 6.142154877301592e-7 |
| Аппаратно декодированный V против сохранённого software decode, relative L2 | 0.0010855946325182854 |
| То же, max absolute difference | 0.0037622451782226562 |
| FP16 K против сохранённых offline values | точное совпадение |

Все три attention-ветки прошли исходный `relative_l2 <= 1e-4` gate. Отдельное расхождение программного и аппаратного ASTC decode обнаружено, но его причина и влияние на модель здесь ещё не установлены. В исходном коде это сравнение только выводилось и не определяло прохождение gate. Не следует называть software и hardware decode тождественными или считать прежнюю модельную оценку автоматически аппаратно воспроизведённой.

| Учёт памяти первого fixture | Байты |
|---|---:|
| ASTC V payload | 33 792 |
| ASTC V texture `allocatedSize` | 49 152 |
| FP16 K | 147 456 |
| Metadata V | 4 608 |
| K + V + metadata, логически | 185 856 |
| K + V + metadata, сумма выделений | 201 216 |
| Compressed upload staging | 49 152 |

Сумма выделений кэша не включает scratch, dummy bindings, все диагностические копии и временные ресурсы. Реальное выделение packed U4 на этом устройстве здесь не измерено. `production_peak_bytes` остаётся `null`. Один раунд timings сохранён как диагностика и не используется для утверждений об ускорении или пропускной способности памяти.

## Исправления запуска и выполненные проверки

`metal/build.sh` теперь всегда сначала собирает C++ host с доступным CLT SDK. При наличии `xcrun metal` он сохраняет офлайн-ветку `.air` -> `.metallib`. Если утилита отсутствует, записывает путь к исходному `.metal` для runtime compilation. Выбранный путь сохраняется в `shader_path.txt`; оставшийся от прошлой сборки `.metallib` не выбирается случайно. Успешная сборка host сама по себе не обозначает успешную компиляцию шейдера.

`scripts/run_mac_probe.sh` принимает:

- `PYTHON` - интерпретатор с NumPy для summary; его версия отдельно сохраняется в `summary_python.json`;
- `OUT_DIR` - новый каталог результатов; при существующем пути запуск отклоняется до сборки и GPU;
- `ASTC_KV_METAL_BUILD_DIR` - необязательный каталог сборки, по умолчанию `OUT_DIR/build`.

`probe_environment.sh` продолжает снимать системный `python3` отдельно; поле его зависимостей не следует считать версией выбранного summary-интерпретатора. Ошибки каждого GPU fixture сохраняются в отдельных `.stderr`.

Пример следующей серии (новое имя OUT_DIR обязательно):

```bash
PYTHON=/home/researcher/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  OUT_DIR=results/mac_session_01 bash scripts/run_mac_probe.sh
```

Проверено без новых GPU-запусков:

```bash
bash -n metal/build.sh scripts/run_mac_probe.sh
ASTC_KV_METAL_BUILD_DIR=results/revalidation_20261028/metal/launcher_build \
  bash metal/build.sh
OUT_DIR=results/revalidation_20261028/metal bash scripts/run_mac_probe.sh
```

Первые две команды прошли. Третья намеренно получила exit 2 и сообщение `Refusing to overwrite existing results`; ни GPU, ни перезапись результатов не выполнялись. Итог - `launcher_validation.json`, сборка - `launcher_build.log`. Ветка офлайн-компилятора не выполнена, поскольку такого компилятора на машине нет.

## Состояние передачи следующего запуска

Был подготовлен `astc-kv-probe-readback` с сохранением аппаратно декодированных K/V и исправлениями проверки конечности reference/warmup. Запрос разрешения на два GPU readback запуска ожидал примерно 1136 секунд и затем был прерван. Это ожидание разрешения, **не аппаратная ошибка**; выполнение этих двух запусков на момент передачи основному агенту не подтверждено. Основной агент продолжает GPU-проверки и дополнит документ новыми данными.

Остаются: установить причину decode discrepancy; проверить оба fixtures и граничные формы; отдельно проверить корректность полного CPU/GPU decode сравнения и влияние округления; затем провести повторяемые диагностические timings с согласованными входами и границами времени. Сильный runtime baseline, динамическое кодирование и полная генерация остаются отдельными этапами.

## Завершение основным потоком

После передачи оба readback запуска выполнены: `readback_v.json`, `readback_kv.json`
и FP32 файлы в одноимённых подкаталогах. CPU-анализ сохранён в
`decode_comparison.json`. Затем введён отдельный инженерный decode smoke gate;
он калиброван после первоначальных измерений и не удостоверяет качество модели.
`final_controls/validation.json` фиксирует 7/7 ожидаемых исходов: четыре успешных
storage cases и три отказа до timing (gross wrong, nonfinite и missing manifest field).

Полный launcher выполнен командой:

```sh
PYTHON=.venv-research/bin/python OUT_DIR=results/revalidation_20261028/metal/launcher_final bash scripts/run_mac_probe.sh
```

Он завершился exit0. В `launcher_final/` сохранены build log, окружение, оба
correctness результата, 30 rounds * 10 iterations * 3 variants для каждого
fixture и обе timing summaries. V-only GPU medians: ASTC 81,035 мкс,
ordinary texture 82,137 мкс, FP16 buffer 43,281 мкс. Это один малый
диагностический working set и простые ядра; вывода об inference speedup нет.
Полная причина различий decode, остальные формы, сильный fused Q4 comparison,
динамический encoder и runtime по-прежнему требуют отдельных опытов.
