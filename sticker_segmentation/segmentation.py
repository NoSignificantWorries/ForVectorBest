import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# Загрузка вашей обученной модели YOLOv8
model = YOLO('runs/segment/train2/weights/best.pt')  # Укажите путь к вашей модели

# Функция для обработки каждого изображения
def process_image(image_path):
    # Проверка наличия папки для сохранения результатов
    if not os.path.exists('segmentation_results'):
        os.makedirs('segmentation_results')
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_orig = image.copy()
    h_or, w_or = image.shape[:2]
    image_resized = cv2.resize(image, (640, 640))  # Изменение размера изображения для работы модели
    results = model(image_resized)[0]
    
    if results.masks is None:
        print(f"Для изображения {image_path} не найдено масок.")
        return  # Если нет масок, пропускаем изображение
    
    masks = results.masks.data.cpu().numpy()  # Получаем маски объектов

    # Проверим, сколько масок обнаружено
    print(f"Обнаружено масок: {len(masks)} для изображения {image_path}")

    # Наложение масок на изображение
    for i, mask in enumerate(masks):
        # Поскольку у нас только один класс, мы можем использовать один цвет для маски
        color = (0, 255, 0)  # Зеленый цвет для маски (можно выбрать любой)

        # Изменение размера маски перед созданием цветной маски
        mask_resized = cv2.resize(mask, (w_or, h_or))
        
        # Создание цветной маски
        color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
        color_mask[mask_resized > 0] = color

        # Сохранение маски в файл с уникальным именем
        mask_filename = os.path.join('segmentation_results', f"{os.path.splitext(os.path.basename(image_path))[0]}_mask_{i}.png")
        cv2.imwrite(mask_filename, color_mask)
        print(f"Сохранена маска: {mask_filename}")

        # Наложение маски на исходное изображение
        image_orig = cv2.addWeighted(image_orig, 1.0, color_mask, 0.5, 0)

    # Сохранение измененного изображения
    new_image_path = os.path.join('segmentation_results', os.path.splitext(os.path.basename(image_path))[0] + '_segmented' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, image_orig)
    print(f"Segmented image saved to {new_image_path}")

# Обработка всех изображений в папке
def process_images_in_folder(folder_path):
    # Получаем все изображения в папке
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpeg'))
    
    # Обработка каждого изображения
    for image_path in image_paths:
        print(f"Обрабатываем изображение: {image_path}")
        process_image(image_path)

# Пример использования: укажите путь к папке с изображениями
process_images_in_folder('YOLO_dataset/test/images')  # Замените на путь к вашей папке с изображениями
