import json
import os
import shutil

# Путь к твоему датасету
dataset_path = "/mnt/c/My_files/Study/ForVectorBest2/Stickers_dataset"
output_path = "/mnt/c/My_files/Study/ForVectorBest2/YOLO_dataset"

# Список папок (train, valid, test)
folders = ["train", "valid", "test"]

# Проходим по каждой из папок
for folder in folders:
    coco_file = os.path.join(dataset_path, folder, "_annotations.coco.json")
    
    # Проверяем, существует ли файл аннотаций для этой папки
    if not os.path.exists(coco_file):
        print(f"Файл аннотаций для {folder} не найден: {coco_file}")
        continue
    
    print(f"Обрабатываем папку: {folder}")
    
    # Читаем файл COCO для текущей папки
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Создаем папки для аннотированных данных YOLOv8
    output_folder = os.path.join(output_path, folder)
    output_image_folder = os.path.join(output_folder, "images")
    output_label_folder = os.path.join(output_folder, "labels")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    
    # Копируем изображения в папку 'images'
    image_folder = os.path.join(dataset_path, folder)
    
    for image in coco_data['images']:
        image_filename = image['file_name']
        image_path = os.path.join(image_folder, image_filename)
        shutil.copy(image_path, os.path.join(output_image_folder, image_filename))

    # Конвертируем аннотации в формат YOLOv8
    categories = coco_data['categories']
    category_dict = {category['id']: category['name'] for category in categories}

    for image in coco_data['images']:
        image_id = image['id']
        image_filename = image['file_name']
        image_width = image['width']
        image_height = image['height']

        # Получаем аннотации для изображения
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        # Создаем файл аннотаций для изображения в формате YOLO
        yolo_annotation_filename = os.path.join(output_label_folder, image_filename.replace('.jpg', '.txt'))

        # Проверка, существует ли файл аннотации
        if os.path.exists(yolo_annotation_filename):
            print(f"Аннотация для {image_filename} уже существует, пропускаем...")
            continue

        with open(yolo_annotation_filename, 'w') as f:
            for annotation in annotations:
                # Заменим category_id на 0 (поскольку у вас только один класс)
                category_id = 0  # Это должно быть 0, так как у вас только один класс
                category_name = category_dict[annotation['category_id']]  # Мы сохраняем название категории для справки, но не используем
                segmentation = annotation['segmentation']  # Список полигонов

                for poly in segmentation:
                    # Преобразуем полигон в формат YOLO (нормализуем)
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        x = poly[i] / image_width
                        y = poly[i+1] / image_height
                        normalized_poly.append(x)
                        normalized_poly.append(y)

                    # Записываем полигон в файл (YOLO использует формат: category_id followed by points)
                    # YOLOv8 для сегментации требует все полигоны одного объекта в одной строке
                    poly_str = ' '.join(map(str, normalized_poly))
                    f.write(f"{category_id} {poly_str}\n")

    print(f"Конвертация для {folder} завершена!")

print("Все конверсии завершены!")
