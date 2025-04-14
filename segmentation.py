import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('runs/segment/train2/weights/best.pt')

def process_image(image_path):
    if not os.path.exists('segmentation_results'):
        os.makedirs('segmentation_results')
    
    image = cv2.imread(image_path)
    image_orig = image.copy()
    h_or, w_or = image.shape[:2]
    image_resized = cv2.resize(image, (640, 640))
    results = model(image_resized)[0]
    
    if results.masks is None:
        print(f"Для изображения {image_path} не найдено масок.")
        return
    
    masks = results.masks.data.cpu().numpy()
    print(f"Обнаружено масок: {len(masks)} для изображения {image_path}")

    for i, mask in enumerate(masks):
        color = (0, 255, 0)
        
        mask_resized = cv2.resize(mask, (w_or, h_or))
        
        color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
        color_mask[mask_resized > 0] = color

        mask_filename = os.path.join('segmentation_results', f"{os.path.splitext(os.path.basename(image_path))[0]}_mask_{i}.png")
        cv2.imwrite(mask_filename, color_mask)
        print(f"Сохранена маска: {mask_filename}")

        image_orig = cv2.addWeighted(image_orig, 1.0, color_mask, 0.5, 0)

    new_image_path = os.path.join('segmentation_results', os.path.splitext(os.path.basename(image_path))[0] + '_segmented' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, image_orig)
    print(f"Segmented image saved to {new_image_path}")

def process_images_in_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpeg'))
    
    for image_path in image_paths:
        print(f"Обрабатываем изображение: {image_path}")
        process_image(image_path)

process_images_in_folder('YOLO_dataset/test/images')
