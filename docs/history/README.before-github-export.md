# ASTC-KV

Исследование активного KV-кэша для аппаратного ASTC-декодирования на Apple GPU.
Дата первого пилота: **28 октября 2026**.

**Это исследовательский стенд, а не готовый ускоритель inference.**
Независимая перепроверка 28 октября: все 936 CPU snapshot cases повторены на
Mac; модельный опыт на всех 30 слоях воспроизвёл прежние основные выводы.
Прямое ASTC-чтение в Metal attention впервые в этом проекте выполнено и проверено на M5 Pro.
Динамический GPU-кодировщик, интеграция с runtime и преимущество полной
генерации ещё не подтверждены.

**Актуальный итог:** [проверка адресации и оформление отчетов](docs/research_v3_report.md).
Новые варианты адресации компактного атласа прошли проверки точности.
Первая серия дала около 9.3% выигрыша на T8192, но дополнительная ABBA-серия
не подтвердила условие "не менее 5% во всех пяти сеансах". Память не изменилась.
Отчеты приведены к обычной пунктуации, исходные версии сохранены. Для проверки
старых SHA после правки отчетов: `.venv-research/bin/python scripts/verify_historical_reports.py`.

Предыдущий этап: [продолжение исследования v2](docs/research_v2_report.md).
Новое D64 Metal-ядро снизило GPU time ASTC на T384 до 7,9 us
(исходный tile256: 81 us; настроенный v1: 12 us), но собственный FP16
остаётся быстрее. Перестановка ASTC-блока улучшила snapshot error и
не подтвердила улучшение PPL на четырёх новых текстах. Реальный fused
Metal baseline llama.cpp выполнен после независимой проверки precision-контракта;
сильного преимущества ASTC не найдено. Новизна и прорыв не установлены.
Компактный атлас тех же блоков прошёл побитную проверку hardware decode:
на T8192 V texture allocation снизился на 53,125%, persistent K+V+metadata -
на 17,35% против прежнего ASTC layout, но GPU latency выросла примерно на 14,5%.
Это проверенный обмен памяти на время на повторённых блоках, не ускорение модели.
Raw: `results/kernel_v2/`, `results/block_geometry_v2/`,
`results/geometry_holdout_v2/`. Архив и прежние результаты сохранены.

Предыдущий этап: [независимая перепроверка](docs/revalidation_report_20261028.md).
Комбинация U4 K + ASTC V: PPL +1,3564% к FP16, 1,7192% экономии
логических байтов к asymmetric U4. Поправка среднего смещения V не прошла
проверку; исправлено изменение общего Q в MLX-стенде. Этот опыт не следует
смешивать с новым V-only geometry holdout.

## Читать сначала

* `docs/initial_report.md` - результаты, решения и ограничения первого пилота.
* `docs/prior_art.md` / `.json` - проверенные первичные работы и граница новизны.
* `docs/protocol.md` - зафиксированные критерии качества и протокол измерений.
* `docs/representation_v0.1.md` - представление, адресация и точный учёт битов.
* `docs/metal_path.md` - аппаратный путь и ограничения неподтверждённого прототипа.
* `docs/followup_experiments.md` - следующие различающие гипотезы эксперименты.
* `results/experiment_log.jsonl` - журнал, включая отрицательные результаты.
* `results/revalidation_20261028/experiment_log.jsonl` - журнал продолжения.
* `docs/independent_research_20261028.md` - обновлённая независимая проверка prior art.
* `docs/research_v2_report.md` - новые GPU-ядра, реальный fused baseline и свежий PPL holdout.
* `docs/research_decision_v2.md` - независимый разбор решения о продолжении ветки.
* `docs/master_prompt_v0.1_ru.md` - неизменённая копия исходной постановки.

## Самый короткий следующий шаг на Mac

В архив уже входят настоящие ASTC-текстуры и запросы выбранного слоя. Для
первого аппаратного теста не нужно скачивать модель или собирать CPU-кодек.
Нужны Apple Silicon Mac, Command Line Tools SDK и Python с NumPy.
При отсутствии офлайн `metal` compiler используется runtime MSL compilation:

```bash
cd astc-kv
PYTHON=.venv-research/bin/python OUT_DIR=results/mac_new bash scripts/run_mac_probe.sh
```

Скрипт сохраняет окружение, лог сборки, проверку hardware decode/attention,
сырые измерения и сводку в выбранном новом каталоге; существующий каталог
он отказывается перезаписывать. Укажите установленный Python с NumPy;
`.venv-research` - окружение текущей перепроверки, не часть исходного ZIP.
В песочнице Metal может быть недоступен: запуску нужен разрешённый доступ к GPU.
Маленький supplied fixture предназначен для корректности и диагностики,
не для заявления об ускорении длинного контекста.

## Воспроизвести численный пилот

Python3.12+, компилятор C++17 и около 2GB свободного места для зависимостей и
промежуточных данных. Linux и macOS CPU bridge выполнены и проверены.
Веса и upstream код загружаются по зафиксированным
ревизиям с проверкой SHA256; они не включены в архив проекта.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_dependencies.py
bash scripts/build_codec.sh
python -m unittest discover -s tests -v
```

Настоящие captures уже сохранены в `results/captures/`; каждый содержит
Q/K после RoPE, V, token IDs, provenance и диагностические данные. Их можно
восстановить из исходных весов:

```bash
OPENBLAS_NUM_THREADS=2 python scripts/capture_qkv.py \
  --model-dir third_party/model --text-file data/prompts/systems.txt \
  --out results/new_capture_systems.npz --max-tokens 384 \
  --layers 0,5,11,17,23,29 --split tune --include-pre-rope
```

Прогон фиксированного списка вариантов:

```bash
mkdir -p results/reproduction
for name in systems story science records; do
  OPENBLAS_NUM_THREADS=1 python scripts/run_pilot.py \
    --capture "results/captures/$name.npz" \
    --out "results/reproduction/pilot_$name.jsonl"
done
python scripts/summarize_pilot.py results/reproduction/pilot_*.jsonl \
  --out results/reproduction/pilot_summary.json
```

Полная причинная CPU-проверка продолжения по всем 30 слоям:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/evaluate_model.py \
  --out-prefix results/reproduction/model_quality
```

Она сначала делает чистый prefill288, сжимает завершённый старый префикс,
затем вычисляет реальные новые K/V по изменённой истории. Никакая статистика
будущих токенов не используется. CPU хранит восстановленные массивы; этот
прогон не измеряет физическую память сжатого GPU-кэша или скорость inference.

Независимая проверка уже выполнена на PyTorch2.6.0+cpu / Transformers4.51.3.
Для повторения дополнительно установите `torch==2.6.0`,
`transformers==4.51.3`, `safetensors==0.5.3` и добавьте
`--validate-transformers` к capture-команде. На Linux используйте CPU-колесо
PyTorch, если CUDA не требуется. Результаты исходной сверки находятся в
`results/transformers_validation.json`.

## Сильные базовые варианты на Mac

`scripts/probe_mlx.py` содержит изолированный cached-attention benchmark
реальных `mx.fast.scaled_dot_product_attention` и MLX-LM quantized KV attention.
Он требует установленного MLX/MLX-LM на Mac и сохраняет фактические версии.
Точная CLI доступна через `python scripts/probe_mlx.py --help`.
Это ещё не benchmark полной генерации или динамического обновления кэша.

Сравнение также должно включать актуальный llama.cpp Metal FlashAttention
с Q4_0/Q8_0 KV на той же модели и фиксированных весах. Open-TQ-Metal - ещё один
первичный ориентир, требующий проверки его кода, D=128 ограничения и baseline.
Заявленные в чужой статье ускорения не используются как наши измерения.

## Структура

```text
astc_kv/       настоящий float32 bridge к Arm astcenc, представления и метрики
scripts/       загрузка, capture, эксперименты, model quality, экспорт на Mac
metal/         Objective-C++ host и Metal attention/diagnostic kernels
configs/       первоначальная конфигурация и критерии до получения результатов
tests/         адресация, GQA, диапазоны, размеры и настоящий ASTC roundtrip
data/prompts/  четыре оригинальных диагностических текста
results/       сырые данные, captures, provenance, результаты и fixtures
docs/          обзор, спецификация, протокол, отчёт и следующие эксперименты
```

## Предел текущего результата

Четыре авторских англоязычных текста и 384 токена - пилот. Они не заменяют
репрезентативный публичный benchmark, длинный контекст, вторую модель,
прикладные задачи или оценку качества свободной генерации. Первое представление
может пройти диагностическую проверку attention и не пройти модельный порог.
Отрицательные результаты сохраняются. Обнаружение нового принципа и полезное
ускорение пока не установлены.

Upstream ASTC encoder: Arm, Apache-2.0. Model: HuggingFaceTB/SmolLM2-135M,
Apache-2.0. Их собственные лицензии сохраняются в скачиваемых исходниках и
официальных репозиториях; проект не включает сторонние веса/сборки в deliverable.
