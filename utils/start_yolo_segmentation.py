import os
import argparse

import ultralytics as ult

# custom modules
from src.segmentator.segmentation import YOLOSegmentation
import src.conf as conf


def main(mode: str, data_path: str, epochs: int, imgsz: int, model_type: str, weights: str, source: str) -> None:
    conf.DEBUG_OUTPUT = True
    if mode == 'train':
        if not data_path:
            raise ValueError("Укажите путь к --data.yaml для обучения")
        model = YOLOSegmentation(weights=model_type)
        model.train_model(data_path, epochs=epochs, imgsz=imgsz)

    elif mode == 'predict':
        if not weights or not source:
            raise ValueError("Для предсказания укажите --weights и --source")
        model = YOLOSegmentation(weights=weights)
        model.predict(source, imgsz=imgsz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8: обучение и предсказание")
    parser.add_argument("-m", '--mode', type=str, required=True, choices=['train', 'predict'], help='Режим: train или predict')

    parser.add_argument("-d", '--data', type=str, help='Путь к data.yaml для обучения')
    parser.add_argument("-e", '--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument("-i", '--imgsz', type=int, default=640, help='Размер изображений')
    parser.add_argument("-t", '--model_type', type=str, default='yolov8n-seg.pt', help='Базовая модель YOLO (для обучения)')

    parser.add_argument("-w", '--weights', type=str, help='Путь к .pt файлу модели')
    parser.add_argument("-s", '--source', type=str, help='Изображение или папка для предсказания')

    args = parser.parse_args()

    main(args.mode, args.data, args.epochs, args.imgsz, args.model_type, args.weights, args.source)
