import json
import os

import shutil

dataset_path = "Stickers_dataset"
output_path = "YOLO_dataset"

folders = ["train", "valid", "test"]

for folder in folders:
    coco_file = os.path.join(dataset_path, folder, "_annotations.coco.json")
    
    if not os.path.exists(coco_file):
        print(f"Файл аннотаций для {folder} не найден: {coco_file}")
        continue
    
    print(f"Обрабатываем папку: {folder}")
    
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    output_folder = os.path.join(output_path, folder)
    output_image_folder = os.path.join(output_folder, "images")
    output_label_folder = os.path.join(output_folder, "labels")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    
    image_folder = os.path.join(dataset_path, folder)
    
    for image in coco_data['images']:
        image_filename = image['file_name']
        image_path = os.path.join(image_folder, image_filename)
        shutil.copy(image_path, os.path.join(output_image_folder, image_filename))

    categories = coco_data['categories']
    category_dict = {category['id']: category['name'] for category in categories}

    for image in coco_data['images']:
        image_id = image['id']
        image_filename = image['file_name']
        image_width = image['width']
        image_height = image['height']

        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        yolo_annotation_filename = os.path.join(output_label_folder, image_filename.replace('.jpg', '.txt'))

        if os.path.exists(yolo_annotation_filename):
            print(f"Аннотация для {image_filename} уже существует, пропускаем...")
            continue

        with open(yolo_annotation_filename, 'w') as f:
            for annotation in annotations:
                category_id = 0
                category_name = category_dict[annotation['category_id']]
                segmentation = annotation['segmentation']

                for poly in segmentation:
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        x = poly[i] / image_width
                        y = poly[i+1] / image_height
                        normalized_poly.append(x)
                        normalized_poly.append(y)

                    poly_str = ' '.join(map(str, normalized_poly))
                    f.write(f"{category_id} {poly_str}\n")

    print(f"Конвертация для {folder} завершена!")

print("Все конверсии завершены!")
