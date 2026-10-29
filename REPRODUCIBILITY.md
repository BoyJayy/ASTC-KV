# Воспроизведение

Команды выполняются из корня распакованного репозитория. Для новых экспериментов всегда используйте новое имя output directory. Сохраненные результаты не перезаписываются.

## 1. Проверка опубликованных файлов без GPU и установки пакетов

Нужен Python 3.12 или новее. Эти команды используют стандартную библиотеку:

```sh
python3 tools/verify_snapshot.py
python3 tools/check_results.py
```

Первая сверяет SHA256 и размеры файлов публичного пакета. Вторая заново считает ABBA-отношения из raw JSON последних трех серий, проверяет численные gates и воспроизводит FAIL всех трех performance-критериев. Это проверка сохраненных измерений, а не новый GPU-запуск.

`release/source_export.json` содержит исходные и экспортные hashes каждого перенесенного файла. Пути рабочей станции в текстах заменены общими примерами. Исторические manifests в `results/` сохранены как записи прежних опытов; их hashes относятся к прежним байтам, а часть локальных бинарников в пакет не входит.

Поэтому `scripts/freeze_research_v*.py --verify`, `scripts/verify_historical_reports.py` и старый `scripts/verify_research_v4.py` являются архивными проверками исходной рабочей станции. Их не следует использовать как проверку этого GitHub-экспорта. Для него предназначены две команды выше. Их переносимая проверка не утверждает, что старые локальные binaries заново собраны с тем же hash.

## 2. Численные тесты без native ASTC

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5
python tools/run_unit_tests.py
```

Одна проверка настоящего astcenc явно пропускается, пока native bridge не собран. Остальные проверяют адресацию, диапазоны, softmax/GQA, парные загрузки и отбраковку неверно размеченных измерений. Пропуск native теста не считается его успешным выполнением.

## 3. Полные CPU-тесты с настоящим astcenc

Нужны C++17 compiler и CMake. На macOS используется Command Line Tools. Зависимости скачиваются только при явном выполнении следующей команды; веса для тестов не нужны.

```sh
python -m pip install -r requirements.txt
python scripts/download_dependencies.py --skip-model
bash scripts/build_codec.sh
python tools/run_unit_tests.py --with-codec
```

## 4. Metal на готовых fixtures

Нужен Apple Silicon Mac с поддержкой ASTC и Command Line Tools. Исторические результаты получены на M5 Pro, 20 GPU cores, 48 GB, macOS 26.6.2. Наличие другого Apple GPU не гарантирует те же времена.

```sh
bash metal/build_atlas_v7.sh
build/metal-atlas-v7/astc-kv-probe-atlas-v7 \
  results/kernel_v2/fixtures_page/t384/manifest.json \
  metal/attention_atlas_v7.metal 30 10 32 - 20261029 \
  > new-metal-result.json
```

При отсутствии offline Metal compiler host компилирует MSL во время запуска. Host build сам по себе не доказывает доступность GPU или корректность shader. Песочница может ограничивать доступ к Metal.

Полные последние протоколы:

```sh
bash metal/build_gather_v8.sh
bash metal/build_gather_exchange_v9.sh
bash metal/build_gqa_v10.sh
python scripts/run_gather_v8.py --out results/new_gather
python scripts/run_gather_exchange_v9.py --out results/new_exchange
python scripts/run_gqa_v10.py --out results/new_gqa
```

Они фиксируют уже экспортированные исходники и входы для нового запуска. Ни выбор числа повторений после просмотра результата, ни замена primary на удачный secondary не входят в протокол. T2048/T8192 повторяют блоки одного сохраненного snapshot и не являются проверкой качества на реальном длинном тексте.

## 5. Полная модельная проверка

```sh
python scripts/download_dependencies.py
bash scripts/build_codec.sh
OPENBLAS_NUM_THREADS=1 python scripts/evaluate_model.py \
  --out-prefix results/new_model_quality
```

Скачивается зафиксированная версия SmolLM2-135M с SHA256-проверками. Этот CPU-опыт моделирует сжатие старого prefix и дальнейшие вычисления с обратной связью ошибок; он не измеряет скорость полного GPU inference. Вторая модель и рабочий динамический ASTC-кэш в этот снимок не входят.

Методика, остальные команды и границы выводов находятся в [отчетах v2](docs/research_v2_report.md), [v3](docs/research_v3_report.md) и [v4](docs/research_v4_report.md). Их команды сохранены исторически; `.venv-research/bin/python` можно заменить своим Python. Для целостности публичного экспорта используйте `tools/verify_snapshot.py`.
