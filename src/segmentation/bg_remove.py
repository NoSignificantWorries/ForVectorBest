import os
import glob

import cv2
import numpy as np

segmentation_results_folder = 'segmentation_results'
bg_remove_results_folder = 'bg_remove_results'


def process_image(image_path):
    if not os.path.exists(bg_remove_results_folder):
        os.makedirs(bg_remove_results_folder)
    
    image = np.float64(cv2.imread(image_path))
    image /= 255
    height, width = image.shape[:2]

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    mask_files = glob.glob(os.path.join(segmentation_results_folder, f'{image_name}_mask_*.png'))
    if not mask_files:
        print(f"WARN: Not found mask for {image_path}.")
        return
    
    for mask_file in mask_files:
        mask_origin = np.float64(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE))
        mask = cv2.resize(mask_origin, (width, height))
    
    mask /= np.max(mask)
    while_pixels = np.argwhere(mask == 1.0)
    y_min, x_min = while_pixels.min(axis=0)
    y_max, x_max = while_pixels.max(axis=0)

    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    cropped_image_with_bg_removed = np.ones_like(cropped_image, dtype=np.float64)
    cropped_image_with_bg_removed[cropped_mask > 0] = cropped_image[cropped_mask > 0]

    colors, counts = np.unique(cropped_image_with_bg_removed.reshape(-1, cropped_image_with_bg_removed.shape[-1]), axis=0, return_counts=True)
    sort_indexes = np.argsort(counts)[::-1]
    sorted_colors = colors[sort_indexes]
    sorted_counts = counts[sort_indexes]
    
    cropped_image_with_bg_removed[cropped_mask < 1] = sorted_colors[3]

    image_to_save = np.uint8(np.round(cropped_image_with_bg_removed * 255))
    new_image_path = os.path.join(bg_remove_results_folder, image_name + '_bg_removed' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, image_to_save)
    print(f"Image with background removed saved to {new_image_path}")


if __name__ == "__main__":
    images_folder = "YOLO_dataset/test/images"

    for image_path in os.listdir(images_folder):
        process_image(f"{images_folder}/{image_path}")

