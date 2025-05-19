# src/

В папке `src` содержатся основные модули и компоненты проекта, реализующие архитектуру и логику обработки данных: от конфигурации и базовых интерфейсов до детекции, предобработки изображений и организации пайплайна.

---

## Содержание

* [base\_worker.py](#base_worker)
* [conf.py](#conf)
* [main.py](#main)
* [detector/](#detector)
* [image\_processor/](#image_processor)
* [pipeline/](#pipeline)
* [Взаимодействие компонентов](#component-interaction)
* [Диаграмма классов (схема)](#class-diagram)
* [Рекомендации по расширению](#recommendations)

---

### `base_worker.py`

Базовый класс `BaseWorker` для всех рабочих компонентов (workers) проекта.

Определяет единый интерфейс, который должны реализовать все наследники:

* `__call__` — основной метод обработки (например, инференс, преобразование и т.д.).
* `verify` — метод валидации результата (например, проверка формата или структуры).
* `save_call` — сохранение результата вызова (например, на диск).

Используется для стандартизации взаимодействия между компонентами (детектор, препроцессор и др.).

---

### `conf.py`

Централизованный конфигурационный файл проекта.

Содержит:

* Пути к данным, весам моделей, директориям сохранения:

  * `BASE_DATASET_PATH`, `DETECTOR_DATASET_PATH`, `SAVE_DIR`, `DETECTOR_WEIGHTS_PATH` и др.
* Параметры инференса и обучения YOLO:

  * `DETECTOR_PARAMS`: `epochs`, `optimizer`, `flipud`, `conf`, `iou` и т.д.
* Классы для детекции (`BBOX_CLASSES`) и визуальные цвета для аннотаций (`CLASS_COLORS`).
* Пути к классификаторам (`BBOX_CLASSIFIER_PATH`, `PATTERN_CLASSIFIER_PATH`).
* Глобальные флаги: `DEBUG_OUTPUT`, `SAVE_MODE`.
* 
---

### `main.py`

Главная точка входа в проект.

* Запускает pipeline обработки.
* Связывает ключевые модули (детектор, препроцессор, и т.д.).
* Управляет:

  * Загрузкой конфигурации.
  * Обработкой входных данных.
  * Сохранением результатов и логикой отладки.

---

### `detector/`

Модуль для работы с объектной детекцией.

* **`detector.py`** — класс `Detector`:

  * Загрузка YOLO модели.
  * Инференс по изображению или батчу.
  * Обучение модели (`train()`).
  * Визуализация результатов (`visualize()`).
  * Сохранение предсказаний (`predict()`).

---

### `image_processor/`

Модуль для предобработки изображений.

* **`image_processor.py`** — класс `Preprocessor`:

  * Преобразование изображений (`__call__`).
  * Сохранение и визуализация результатов (`visualize()`).

---

### `pipeline/`

Организация последовательной обработки:

* **`pipeline.py`** — определяет и запускает цепочку:

  * Связывает все компоненты.
  * Управляет передачей данных.
  * Поддерживает расширение и настройку.

---
<a name="component-interaction"></a>
## Взаимодействие компонентов

* `main.py` инициализирует pipeline.
* `pipeline` вызывает компоненты, унаследованные от `BaseWorker`.
* `conf.py` поставляет параметры и пути.
* `detector/`, `image_processor/` и другие модули реализуют обработку.

---
<a name="class-diagram"></a>
## 📊 Диаграмма классов (Mermaid-схема)

```mermaid
---
config:
  theme: redux
  look: classic
  layout: elk
---
classDiagram
    BaseWorker <|-- Detector : implements
    BaseWorker <|-- Preprocessor : implements
    Detector *-- YOLOModel : uses
    Pipeline o-- Detector
    Pipeline o-- Preprocessor
    Main ..> Pipeline : executes
    Conf <.. Main : loads config
    Conf : +str BASE_DATASET_PATH
    Conf : +dict DETECTOR_PARAMS
    Conf : +bool DEBUG_OUTPUT
    BaseWorker : +__call__()
    BaseWorker : +verify()
    BaseWorker : +save_call()
    Detector : +train()
    Detector : +predict()
    Preprocessor : +__call__()
    Preprocessor : +visualize()
```
---
<a name="recommendations"></a>
## Рекомендации по расширению

* Добавляя новые компоненты обработки, наследуйте их от `BaseWorker` и реализуйте:

  * `__call__`, `verify`, `save_call`, `visualize`.
* Все конфигурации выносите в `conf.py`.
* Используйте `pipeline` как точку интеграции любых новых шагов.
* Соблюдайте интерфейс совместимости — это упростит повторное использование кода.
