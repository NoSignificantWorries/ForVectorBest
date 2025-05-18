import cv2
import json
from pathlib import Path
from src.detector import detection


def process_image(img_path, save_dir, yolo_model, json_data, relative_subfolder, allowed_classes=None):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {img_path}")
        return

    detections = yolo_model.predict(source=str(img_path), save_dir="./tmp_predict", allowed_classes=allowed_classes)

    base_save_path = save_dir / relative_subfolder
    base_save_path.mkdir(parents=True, exist_ok=True)

    orig_save_path = base_save_path / img_path.name
    if not orig_save_path.exists():
        cv2.imwrite(str(orig_save_path), image)

    for detection in detections:
        image_path = Path(detection["image_path"]).name
        crops_info = []

        for i, box_info in enumerate(detection.get("text_boxes", [])):
            x1, y1, x2, y2 = box_info["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Пропущен текстовый bbox {i}: некорректные координаты {[x1, y1, x2, y2]}")
                continue

            cropped = image[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h < 20 or w < 20:
                print(f"⚠️ Пропущен текстовый bbox {i}: слишком мал ({w}x{h})")
                continue

            crop_name = f"{img_path.stem}_text_crop_{i}.png"
            crop_path = base_save_path / crop_name
            cv2.imwrite(str(crop_path), cropped)

            crops_info.append({
                "crop_name": crop_name,
                "bbox": [x1, y1, x2, y2],
                "class": 0
            })

        for i, box_info in enumerate(detection.get("object_boxes", [])):
            x1, y1, x2, y2 = box_info["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Пропущен object bbox {i}: некорректные координаты {[x1, y1, x2, y2]}")
                continue

            cropped = image[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h < 20 or w < 20:
                print(f"⚠️ Пропущен object bbox {i}: слишком мал ({w}x{h})")
                continue

            crop_name = f"{img_path.stem}_object_crop_{i}.png"
            crop_path = base_save_path / crop_name
            cv2.imwrite(str(crop_path), cropped)

            crops_info.append({
                "crop_name": crop_name,
                "bbox": [x1, y1, x2, y2],
                "class": 0
            })

        if crops_info:
            height, width = image.shape[:2]
            json_data.append({
                "image_name": image_path,
                "image_size": [width, height],
                "crops": crops_info
            })



def process_folder(input_dir: str, output_dir: str, model_path: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = detection.Detector(model_type=model_path)
    json_data = []

    for subfolder in ['Good', 'Bad']:
        subfolder_path = input_dir / subfolder
        if not subfolder_path.exists():
            print(f"⚠️ Папка {subfolder_path} не найдена.")
            continue

        for img_file in subfolder_path.glob("*.png"):
            process_image(img_file, output_dir, yolo_model, json_data, subfolder)

    json_path = output_dir / "crops_data.json"
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, ensure_ascii=False, indent=4)

    print(f"✅ Обработка завершена. Результат сохранен в: {json_path}")


if __name__ == "__main__":
    input_folder = "data/for_crops"
    output_folder = "output"
    model_path = "resources/weights/detection.pt"

    process_folder(input_folder, output_folder, model_path)
