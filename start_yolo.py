import os
import argparse

from ultralytics import YOLO

def train_model(data_path, model_type='yolov8n-seg.pt', epochs=50, imgsz=640, project='runs/segment', name='train_custom'):
    model = YOLO(model_type)
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        task='segment'
    )
    print(f"Обучение завершено. Результаты сохранены в {os.path.join(project, name)}")


def predict(model_path, source_path, save=True, imgsz=640):
    model = YOLO(model_path)
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


def main():
    parser = argparse.ArgumentParser(description="YOLOv8: обучение и предсказание")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help='Режим: train или predict')

    parser.add_argument('--data', type=str, help='Путь к data.yaml для обучения')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображений')
    parser.add_argument('--model_type', type=str, default='yolov8n-seg.pt', help='Базовая модель YOLO (для обучения)')

    parser.add_argument('--weights', type=str, help='Путь к .pt файлу модели')
    parser.add_argument('--source', type=str, help='Изображение или папка для предсказания')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            raise ValueError("Укажите путь к --data.yaml для обучения")
        train_model(args.data, model_type=args.model_type, epochs=args.epochs, imgsz=args.imgsz)

    elif args.mode == 'predict':
        if not args.weights or not args.source:
            raise ValueError("Для предсказания укажите --weights и --source")
        predict(args.weights, args.source, imgsz=args.imgsz)

if __name__ == '__main__':
    main()
