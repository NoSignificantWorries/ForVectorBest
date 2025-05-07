import os
import argparse

import cv2
from ultralytics import YOLO


def train_model(data, model_type='yolov8n.pt', epochs=150, imgsz=640, freeze=10, batch=2, save=True, plots=True, optimizer='SGD', save_period=5):
    model = YOLO(model_type)
    
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        freeze=freeze,
        batch=batch,
        save=save,
        plots=plots,
        optimizer=optimizer,
        save_period=save_period,
        val=True,
        patience=150,
        warmup_epochs=5,
        degrees=10,
        multi_scale=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
    	device=0,
    )

    # results = model.val()

    model.save("best.pt")


def predict(weights, source, output_dir, imgsz=640):
    from pathlib import Path
    model = YOLO(weights)
    results = model.predict(source, imgsz=imgsz, conf=0.5, iou=0.5)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        im_bgr = r.plot()
        image_path = output_path / f"test_result_{i}.jpg"
        cv2.imwrite(str(image_path), im_bgr)



def main():
    parser = argparse.ArgumentParser(description="YOLO: обучение и предсказание")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help='Режим работы: train для обучения или predict для предсказания')

    parser.add_argument('--data', type=str, help='Путь к data.yaml для обучения')
    parser.add_argument('--epochs', type=int, default=150, help='Количество эпох обучения')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображений для обучения')
    parser.add_argument('--model_type', type=str, default='yolov8n.pt', help='Базовая модель для обучения')
    parser.add_argument('--batch', type=int, default=2, help='Размер батча для обучения')
    parser.add_argument('--save', type=bool, default=True, help='Сохранять модель после обучения')
    parser.add_argument('--plots', type=bool, default=True, help='Строить графики потерь и метрик')
    parser.add_argument('--save_period', type=int, default=5, help='Частота сохранения модели (каждые N эпох)')

    parser.add_argument('--weights', type=str, help='Путь к .pt файлу модели для предсказания')
    parser.add_argument('--source', type=str, help='Путь к изображению или папке для предсказания')
    parser.add_argument('--output_dir', type=str, required=True, help='Папка для сохранения результатов предсказания')


    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            raise ValueError("Для обучения необходимо указать путь к файлу --data.yaml")
        print(f"Обучение модели {args.model_type} на данных {args.data}...")
        train_model(
            args.data,
            model_type=args.model_type,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            save=args.save,
            plots=args.plots,
            save_period=args.save_period
        )

    elif args.mode == 'predict':
        if not args.weights or not args.source:
            raise ValueError("Для предсказания необходимо указать путь к весам (--weights) и исходным данным (--source)")
        print(f"Предсказание с моделью {args.weights} для источника {args.source}...")
        predict(args.weights, args.source, args.output_dir, imgsz=args.imgsz)

if __name__ == '__main__':
    main()
