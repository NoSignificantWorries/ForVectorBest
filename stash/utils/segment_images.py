import os
import glob
import argparse

from tqdm import tqdm
import cv2

# custom modules
from src.segmentator.segmentation import YOLOSegmentation


def process_images_in_folder(model: YOLOSegmentation, folder_path: str, output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = (glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')) 
                + glob.glob(os.path.join(folder_path, '*.jpeg')) + glob.glob(os.path.join(folder_path, '*.PNG')))
    
    with tqdm(image_paths, desc="Обработка изображений") as pbar:
        for image_path in pbar:
            model.predict_image(image_path)
            pred = model.calc_masks()
            new_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_segmented' + os.path.splitext(image_path)[1])
            cv2.imwrite(new_image_path, pred)


def process_image(model: YOLOSegmentation, image_path: str, output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Обрабатываем изображение: {image_path}")
    model.predict_image(image_path)
    pred = model.calc_masks()
    new_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + '_segmented' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, pred)
    print(f"Изображение {image_path} обработано.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 segmentation predicion")
    parser.add_argument("-m", '--mode', type=str, required=True, choices=['one', 'many'], help='Режим: one или many')

    parser.add_argument("-i", '--input', type=str, help='Путь к data.yaml для обучения')
    parser.add_argument("-o", '--output', type=str, help='Путь к data.yaml для обучения')

    args = parser.parse_args()

    model = YOLOSegmentation()
    if args.mode == "one":
        process_image(model, args.input, args.output)
    else:
        process_images_in_folder(model, args.input, args.output)
