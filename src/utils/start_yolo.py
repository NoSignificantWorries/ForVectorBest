import os
import argparse

import ultralytics as ult

def train_model(data_path: str, model_type: str = 'yolov8n-seg.pt', epochs: int = 30, imgsz: int = 640, project: str = 'runs/segment', name: str = 'train_custom') -> None:
    model = ult.YOLO(model_type)
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        task='segment'
    )
    print(f"Обучение завершено. Результаты сохранены в {os.path.join(project, name)}")


def predict(model_path: str, source_path: str, save: bool = True, imgsz: int = 640) -> list:
    model = ult.YOLO(model_path)
    results = model.predict(
        source=source_path,
        save=save,
        imgsz=imgsz,
        show=False,
        hide_labels=True,
        hide_conf=True,
        line_thickness=0,
        task='segment'
    )
    print(f"Предсказание завершено для: {source_path}")
    return results


def main(mode: str, data_path: str, epochs: int, imgsz: int, model_type: str, weights: str, source: str) -> None:
    if mode == 'train':
        if not data_path:
            raise ValueError("Укажите путь к --data.yaml для обучения")
        train_model(data_path, model_type=model_type, epochs=epochs, imgsz=imgsz)

    elif mode == 'predict':
        if not weights or not source:
            raise ValueError("Для предсказания укажите --weights и --source")
        predict(weights, source, imgsz=imgsz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8: обучение и предсказание")
    parser.add_argument("-m", '--mode', type=str, required=True, choices=['train', 'predict'], help='Режим: train или predict')

    parser.add_argument("-d", '--data', type=str, help='Путь к data.yaml для обучения')
    parser.add_argument("-e", '--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument("-s", '--imgsz', type=int, default=640, help='Размер изображений')
    parser.add_argument("-t", '--model_type', type=str, default='yolov8n-seg.pt', help='Базовая модель YOLO (для обучения)')

    parser.add_argument("-w", '--weights', type=str, help='Путь к .pt файлу модели')
    parser.add_argument("-s", '--source', type=str, help='Изображение или папка для предсказания')

    args = parser.parse_args()

    main(args.mode, args.data, args.epochss, args.imgsz, args.model_type, args.weights, args.source)
