import os
import argparse

import cv2
from ultralytics import YOLO

import src.conf as conf
from src.detector import detection


def train_model(data: str, params: dict = conf.DETECTOR_PARAMS) -> None:      
    model = detection.Detector(params["model_type"])
    model.train(data, params)


def predict(source: str, save_dir: str, weights: str = conf.DETECTOR_WEIGHTS_PATH, params: dict = conf.DETECTOR_PARAMS) -> None:
    model = detection.Detector(weights)
    model.predict(source, save_dir, params)


def main():
    parser = argparse.ArgumentParser(description="YOLO: обучение и предсказание")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"],
                        help="Режим работы: train для обучения или predict для предсказания")

    parser.add_argument("--data", type=str, help="Путь к data.yaml для обучения")
    parser.add_argument("--epochs", type=int, default=150, help="Количество эпох обучения")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображений для обучения")
    parser.add_argument("--model_type", type=str, default="yolov8n.pt", help="Базовая модель для обучения")
    parser.add_argument("--batch", type=int, default=2, help="Размер батча для обучения")
    parser.add_argument("--save", type=bool, default=True, help="Сохранять модель после обучения")
    parser.add_argument("--plots", type=bool, default=True, help="Строить графики потерь и метрик")
    parser.add_argument("--save_period", type=int, default=5, help="Частота сохранения модели (каждые N эпох)")

    parser.add_argument("--weights", type=str, help="Путь к .pt файлу модели для предсказания")
    parser.add_argument("--source", type=str, help="Путь к изображению или папке для предсказания")
    parser.add_argument("--output_dir", type=str, required=True, help="Папка для сохранения результатов предсказания")


    args = parser.parse_args()

    if args.mode == "train":
        if not args.data:
            raise ValueError("Для обучения необходимо указать путь к файлу --data.yaml")

        custom_params = conf.DETECTOR_PARAMS.copy()
        custom_params["model_type"] = args.model_type
        custom_params["epochs"] = args.epochs
        custom_params["imgsz"] = args.imgsz
        custom_params["batch"] = args.batch
        custom_params["save"] = args.save
        custom_params["plots"] = args.plots
        custom_params["save_period"] = args.save_period

        print(f"Обучение модели {args.model_type} на данных {args.data}...")
        train_model(args.data, custom_params)

    elif args.mode == "predict":
        if not args.weights or not args.source:
            raise ValueError("Для предсказания необходимо указать путь к весам (--weights) и исходным данным (--source)")
        
        custom_params = conf.DETECTOR_PARAMS.copy()
        custom_params["imgsz"] = args.imgsz

        print(f"Предсказание с моделью {args.weights} для источника {args.source}...")
        predict(args.source, args.output_dir, args.weights, custom_params)

if __name__ == "__main__":
    main()
