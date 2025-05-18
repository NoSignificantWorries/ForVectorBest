import cv2
import numpy as np
from pathlib import Path
import easyocr
import json
from src.detector import detection


def detect_text_and_recognize(img_path: str, yolo_model: detection, save_dir: str, ocr_reader: easyocr.Reader):
    detections = yolo_model.predict(source=img_path, save_dir=save_dir)

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"❌ Не удалось загрузить изображение: {img_path}")

    recognized_text = []

    for detection in detections:
        image_path = detection["image_path"]
        text_boxes = detection["text_boxes"]

        for i, box_info in enumerate(text_boxes):
            x1, y1, x2, y2 = box_info["bbox"]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Пропущен bbox {i}: некорректные координаты {[x1,y1,x2,y2]}")
                continue

            cropped = image[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h < 20 or w < 20:
                print(f"⚠️ Пропущен bbox {i}: слишком мал ({w}x{h})")
                continue

            if len(cropped.shape) == 2 or cropped.shape[2] == 1:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)

            cropped = cv2.convertScaleAbs(cropped, alpha=1.1, beta=20)

            debug_path = Path(save_dir) / f"debug_crop_{i}.png"
            cv2.imwrite(str(debug_path), cropped)

            try:
                ocr_result = ocr_reader.readtext(cropped, detail=1, paragraph=False)
            except Exception as e:
                print(f"❌ Ошибка OCR на bbox {i}: {e}")
                continue

            if ocr_result:
                for bbox_coords, text, confidence in ocr_result:
                    recognized_text.append({
                        "bbox": [x1, y1, x2, y2],
                        "text": text,
                        "confidence": float(confidence)
                    })
            else:
                print(f"ℹ️ OCR не нашел текст на bbox {i}")

    return recognized_text


if __name__ == "__main__":
    img_path = 'data/neg_gray/1.png'
    save_dir = './output'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    yolo_model = detection.Detector(model_type='resources/weights/detection.pt')
    ocr_reader = easyocr.Reader(['ru'])

    recognized_text = detect_text_and_recognize(img_path, yolo_model, save_dir, ocr_reader)

    json_path = Path(save_dir) / "recognized_text.json"
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(recognized_text, jf, ensure_ascii=False, indent=4)


    if recognized_text:
        for item in recognized_text:
            print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}, Bounding Box: {item['bbox']}")
    else:
        print("❗ Текст не найден.")
