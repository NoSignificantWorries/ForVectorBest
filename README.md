# Label Segmentation & Background Removal with YOLOv8

Автоматизированный пайплайн для обнаружения и сегментации этикеток на изображениях с использованием сегментационной модели YOLOv8. Решение включает подготовку данных, обучение модели, применение сегментации и удаление фона объектов.

---

##  Структура проекта

```
.
├── Stickers_dataset/         # Исходный датасет с аннотациями COCO
├── YOLO_dataset/             # Датасет в формате YOLO (images + labels)
├── segmentation_results/     # Маски и изображения с сегментацией
├── bg_remove_results/        # Изображения с удалённым фоном
├── runs/                     # Папка с обученными моделями YOLO
├── my_convert.py             # Конвертация аннотаций COCO → YOLOv8
├── segmentation.py           # Применение сегментации YOLOv8
├── bg_remove.py              # Удаление фона по маскам
├── start_yolo.py             # Скрипт для обучения и предсказания с YOLOv8
├── yolov8n-seg.pt            # Претренированная модель YOLOv8
├── requirements.txt          # Зависимости
└── README.md                 # Описание проекта
```

---

## Установка

### 1. Установите зависимости:

```bash
pip install -r requirements.txt
```

### 2. Установите [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics):

```bash
pip install ultralytics
```

---

## Запуск пайплайна

### 1. Конвертация COCO аннотаций в YOLOv8

Используйте скрипт `my_convert.py` для преобразования аннотаций COCO в формат YOLOv8:

```bash
python my_convert.py
```

### 2. Обучение модели YOLOv8

Для тренировки сегментационной модели используйте скрипт `start_yolo.py`. Пример команды для запуска обучения:

```bash
python start_yolo.py --mode train --data YOLO_dataset/yolov8_config.yaml --epochs 50 --imgsz 640 --model_type yolov8n-seg.pt
```

- **`--mode train`** — режим тренировки.
- **`--data`** — путь к файлу `.yaml`, содержащему конфигурацию данных.
- **`--epochs`** — количество эпох для обучения.
- **`--imgsz`** — размер изображений для обучения.
- **`--model_type`** — базовая модель YOLO для обучения (по умолчанию `yolov8n-seg.pt`).

### 3. Предсказание с обученной моделью

Для предсказания сегментации на новых изображениях или папках с изображениями используйте команду:

- Для предсказания на одном изображении:

```bash
python start_yolo.py --mode predict --weights runs/segment/train2/weights/best.pt --source YOLO_dataset/test/images/IMG_1435_PNG_1_png.rf.b4bdd24bad373f7c578bede2adbdb10e.jpg
```

- Для предсказания на всех изображениях в папке:

```bash
python start_yolo.py --mode predict --weights runs/segment/train2/weights/best.pt --source YOLO_dataset/test/images
```

- **`--mode predict`** — режим предсказания.
- **`--weights`** — путь к файлу модели `.pt`.
- **`--source`** — путь к изображению или папке с изображениями для предсказания.

### 4. Применение сегментации

Для применения сегментации на изображениях используйте скрипт `segmentation.py`. Это наложит маски на изображения и сохранит результаты в папке `segmentation_results/`:

```bash
python segmentation.py
```

Результат: `segmentation_results/`:
- **`*_mask_X.png`** — бинарные маски.
- **`*_segmented.jpg`** — изображения с наложенными масками.

### 5. Удаление фона по маскам

Для удаления фона с объектов используйте скрипт `bg_remove.py`:

```bash
python bg_remove.py
```

Результат: `bg_remove_results/` — изображения с вырезанными объектами и удалённым фоном.

---
