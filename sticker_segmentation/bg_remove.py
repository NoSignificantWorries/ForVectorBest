import cv2
import numpy as np
import os
import glob

# Папка, где сохраняются сгенерированные маски
segmentation_results_folder = 'segmentation_results'
# Папка, где сохраняются результаты удаления фона
bg_remove_results_folder = 'bg_remove_results'

# Функция для обработки каждого изображения
def process_image(image_path):
    # Проверка наличия папки для сохранения результатов
    if not os.path.exists(bg_remove_results_folder):
        os.makedirs(bg_remove_results_folder)
    
    # Загрузка исходного изображения
    image = cv2.imread(image_path)
    image_orig = image.copy()
    h_or, w_or = image.shape[:2]

    # Получаем имя изображения без расширения для поиска соответствующих масок
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Поиск масок для текущего изображения (маски должны иметь имя, соответствующее имени изображения)
    mask_files = glob.glob(os.path.join(segmentation_results_folder, f'{image_name}_mask_*.png'))
    
    # Если масок нет, выводим сообщение
    if not mask_files:
        print(f"Для изображения {image_path} не найдено масок.")
        return
    
    # Создание пустого изображения для сохранения удалённого фона
    image_with_bg_removed = np.ones_like(image, dtype=np.uint8) * 255
    
    # Загружаем и применяем каждую маску для этого изображения
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)  # Загружаем маску как grayscale
        mask_resized = cv2.resize(mask, (w_or, h_or))  # Масштабируем маску до размера изображения

        # Применение маски: если пиксель в маске > 0, то сохраняем пиксель из исходного изображения
        image_with_bg_removed[mask_resized > 0] = image[mask_resized > 0]

    # Сохранение изображения с удалённым фоном в папку bg_remove_results
    new_image_path = os.path.join(bg_remove_results_folder, image_name + '_bg_removed' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, image_with_bg_removed)
    print(f"Image with background removed saved to {new_image_path}")

# Обработка всех изображений в папке
def process_images_in_folder(folder_path):
    # Получаем все изображения в папке (путь к изображениям из папки test/images)
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpeg'))
    
    # Обработка каждого изображения
    for image_path in image_paths:
        print(f"Обрабатываем изображение: {image_path}")
        process_image(image_path)

# Пример использования: укажите путь к папке с изображениями
process_images_in_folder('YOLO_dataset/test/images')  # Замените на путь к вашей папке с изображениями
