> Обновление интегратора: все18 bounded allocations выполнены на M5 Pro. Raw `results/kernel_v2/atlas_allocation.json`; следующий v6 опыт отдельно проверил настоящий upload/decode/attention. Этот allocation probe сам GPU commands не отправляет.

# ASTC atlas allocation-only probe

Мотивация: в текущем T8192 fixture V payload составляет 726528 bytes, а texture allocation — 1572864 bytes (около 2.16×). Новый опыт проверяет, меняется ли размер выделения при размещении того же количества ASTC-блоков в другом atlas. Он не отменяет отрицательный performance result текущей раскладки.

`metal/probe_atlas_alloc.mm` использует только Metal allocation API. Нет shader, command queue, blit, upload, readback, attention, timing или quality evaluation. Успешное выделение texture **не подтверждает** работоспособность нового адресования блоков или эквивалентность decode. `allocatedSize` — размер ресурса по Metal API, не измерение физической residency.

Формат во всех случаях ASTC 6×6 LDR, private, shader-read, без mipmaps, `MTLTextureType2DArray`. Для T384/T2048/T8192 вычисляется `padded=ceil(T/96)*96`, `B=3*11*(padded/6)` исходных блоков по 16 bytes. Формы:

| Размещение | Width | Height | Array length |
|---|---:|---:|---:|
| Исходное | 66 | padded | 3 |
| Головы по горизонтали | 198 | padded | 1 |
| Компактное, W=32/64/128/256 блоков | 6W | 6 ceil(B/W) | 1 |

Каждый JSON case сохраняет B, capacity blocks, exact capacity flag, payload bytes, дополнительные padding blocks/bytes, padding ratio и actual texture `allocatedSize`. Для диагностики также записываются `heapTextureSizeAndAlign` estimate и device allocation до/во время/после scope. Capacity ≥ B — только геометрическая возможность разместить блоки.

Границы: dimension ≤16384; capacity/original blocks ≤1.2, иначе case skipped. Каждый texture создаётся в собственном autoreleasepool и освобождается до следующего. Live-budget 64 MiB проверяется через estimate до allocation и actual `allocatedSize` после; сохраняется baseline device allocation. Все 18 форм малы, shader/driver compilation отсутствует. Case allocation failure или budget skip возвращает ненулевой exit; unsupported не выдаётся за PASS.

Сборка существующим CLT, без установки компонентов и без offline Metal compiler:

```bash
bash -n metal/build_atlas_alloc.sh
bash metal/build_atlas_alloc.sh
```

Статус подготовки: shell syntax и host build PASS; аппаратные allocations агент подготовки не выполнял. Единственная команда запуска для основного агента:

```bash
build/metal-atlas-alloc/astc-kv-atlas-alloc
```

Результат выводится JSON в stdout, ошибки — stderr. Перенаправлять только в новые пути; старые результаты не перезаписывать. Probe не принимает пользовательские dimensions и не изменяет существующие fixtures/source/results. Следующий возможный этап зависит от реальных allocation results и потребует отдельного block remapping + decode correctness, прежде чем какие-либо speed conclusions станут допустимы.
