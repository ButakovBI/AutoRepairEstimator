# `scripts/ml/` — ML training pipeline

Инструменты для подготовки датасета и обучения моделей `parts` и `damages`.

**Полный референс по разметке и гиперпараметрам**:
[`docs/ml/annotation_and_training_guide.md`](../../docs/ml/annotation_and_training_guide.md).

## Структура

| Файл                             | Назначение                                                               |
| -------------------------------- | ------------------------------------------------------------------------ |
| `audit_dataset.py`               | Проверка датасета: orphan-файлы, битые строки, per-class статистика.     |
| `split_dataset.py`               | Стратифицированный split train/val/test в каноническую структуру YOLO.   |
| `train_config.py`                | Единая конфигурация гиперпараметров обучения.                            |
| `train_parts.py`                 | Запуск обучения модели деталей.                                          |
| `train_damages.py`               | Запуск обучения модели повреждений.                                      |
| `inspect_label_distribution.py`  | Лёгкая диагностическая статистика по меткам (подмножество `audit_dataset`). |

## Полный пайплайн (от разметки до весов)

### 1. Разметили в CVAT, экспортировали в YOLO-seg

Положили новые разметки поверх существующих:

* Parts — картинки в `test/images_details/Train/`, метки в `test/labels/Train/`.
* Damages — картинки в `test/0-604damages/images/Train/`, метки в
  `test/0-604damages/labels/Train/`.

### 2. Проверили датасет

```bash
python scripts/ml/audit_dataset.py \
    --images test/images_details/Train \
    --labels test/labels/Train \
    --classes 12 \
    --names door front_fender rear_fender trunk hood roof headlight \
            front_windshield rear_windshield side_window wheel bumper
```

Ожидаем:

* `Malformed lines: 0` — обязательно.
* `Orphan images` — либо 0, либо все перечисленные картинки сознательно
  оставлены без разметки (фон).
* Per-class counts — видим, у кого меньше 30 картинок, запоминаем для
  дозбора.

Если `exit_code != 0` — **не обучать**, сначала чинить датасет.

### 3. Сделали стратифицированный split

**Parts:**

```bash
python scripts/ml/split_dataset.py \
    --images test/images_details/Train \
    --labels test/labels/Train \
    --output test/parts \
    --classes 12 \
    --names door front_fender rear_fender trunk hood roof headlight \
            front_windshield rear_windshield side_window wheel bumper \
    --ratios 0.8 0.1 0.1 \
    --seed 42
```

**Damages** — обязательно с `--oversample` для редких классов:

```bash
python scripts/ml/split_dataset.py \
    --images test/0-604damages/images/Train \
    --labels test/0-604damages/labels/Train \
    --output test/damages \
    --classes 8 \
    --names scratch dent paint_chip rust crack broken_glass flat_tire broken_headlight \
    --ratios 0.8 0.1 0.1 \
    --seed 42 \
    --oversample flat_tire=6 rust=3
```

На выходе — каноническая YOLO-структура:

```
test/parts/
├─ images/
│   ├─ train/  (174 jpg)
│   ├─ val/    (26 jpg)
│   └─ test/   (21 jpg)
├─ labels/
│   ├─ train/  (174 txt)
│   ├─ val/
│   └─ test/
└─ data.yaml
```

`seed=42` зафиксирован специально, чтобы split был **воспроизводимым**. Если
переразметили — перезапускаете ту же команду, сплит пересобирается.

### 4. Загрузили в Colab

```bash
# локально
cd test && zip -r parts.zip parts/
cd test && zip -r damages.zip damages/
```

(В PowerShell: `Compress-Archive -Path test\parts -DestinationPath test\parts.zip`.)

Загрузили `parts.zip` и `damages.zip` на Google Drive / в Colab. В Colab:

```python
!unzip -q /content/drive/MyDrive/parts.zip -d /content/
!unzip -q /content/drive/MyDrive/damages.zip -d /content/
```

Также загрузили всю папку `scripts/ml/` в Colab (через Drive или git clone
репозитория).

В актуальном `split_dataset.py` ключ **`path:` в `data.yaml` не пишется**:
Ultralytics 8.4+ берёт корень датасета как **каталог, в котором лежит сам
`data.yaml`**. После `unzip` в `/content/parts` пути `train: images/train`
и т.д. резолвятся сами — **ничего править не нужно**.

Если у вас старый zip, где в yaml ещё есть строка `path: D:\...` или
`/mnt/d/...`, удалите эту строку (или пересоберите сплит свежим скриптом).
В `colab_train.ipynb` ячейка §3 делает это автоматически.

### 5. Обучили

**Parts** (Colab T4, ~2–4 часа):

```bash
!pip install ultralytics
!python /content/repo/scripts/ml/train_parts.py \
    --data /content/parts/data.yaml \
    --device 0 \
    --batch 8
```

**Damages** (аналогично):

```bash
!python /content/repo/scripts/ml/train_damages.py \
    --data /content/damages/data.yaml \
    --device 0 \
    --batch 8
```

Все гиперпараметры (150 epochs, AdamW, lr=0.001, cls_pw для дисбаланса,
copy_paste=0.3) — из `train_config.py`. Не меняйте через CLI без причины.

### 6. Проверили качество

После обучения в `runs/segment/<run_name>/` лежат:

* `weights/best.pt` — лучший чекпоинт по val mAP. Этот файл едет в прод.
* `results.csv` — метрики по эпохам. Смотрим `metrics/mAP50-95(M)` — на
  плато после ~80 эпох.
* `confusion_matrix.png` — сразу видно, какие классы путаются.
* `BoxPR_curve.png` / `MaskPR_curve.png` — PR-кривые.

**Gate-метрики для выкатки в прод** (минимум для адекватной модели):

| Метрика                         | Parts   | Damages |
| ------------------------------- | ------- | ------- |
| `mAP50(M)` на val (avg)         | ≥ 0.65  | ≥ 0.45  |
| Худший per-class `mAP50(M)`     | ≥ 0.30  | ≥ 0.20  |
| `mAP50(B)` на test              | ≥ 0.60  | ≥ 0.40  |

Если ниже — **в прод не катим**, возвращаемся в CVAT по пунктам раздела 2 в
[`annotation_and_training_guide.md`](../../docs/ml/annotation_and_training_guide.md).

### 7. Подобрали порог уверенности

```python
# быстрый скрипт в Colab — грид по порогам, ищем максимальный F1 на
# ХУДШЕМ per-class (bottleneck, а не среднем):
from ultralytics import YOLO
m = YOLO('runs/segment/damages_.../weights/best.pt')
for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    r = m.val(data='/content/damages/data.yaml', conf=thr, split='val')
    print(f"conf={thr}: mAP50={r.box.map50:.3f}, per-class F1={r.box.f1.tolist()}")
```

Выбрали порог, при котором минимальный `f1[i]` максимален. Записали в
`src/auto_repair_estimator/backend/domain/value_objects/ml_thresholds.py`.

### 8. Выкатили модель

1. Скопировали `best.pt` → `test/details<YYYYMMDD>_learned/best.pt` или
   `test/damages<YYYYMMDD>_learned/best.pt`.
2. Обновили путь в `docker-compose.yml` / `MLWorkerConfig`.
3. Прогнали `pytest tests/integration/ml_worker/` — убедились, что пайплайн
   не сломан на уровне типов и API.
4. Деплой.

## Когда нужно что-то переразметить и запустить заново

* Добавили новых картинок → повторили шаги 2 → 5.
* Поменяли стиль разметки (ярче всего для `scratch`/`crack`) — запустили
  шаги 2 → 5 **на всём датасете**, не только на новом.
* Поменяли список классов в `PartType` / `DamageType` → пересогласовали
  с `data.yaml` **и** с кодом в `src/`. Нельзя менять только в одном
  месте.
