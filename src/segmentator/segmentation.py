import os
import glob

import cv2
import numpy as np
from ultralytics import YOLO

# custom modules
import src.conf as conf


class YOLOSegmentation:
    def __init__(self,
                 save_mode: bool = conf.SAVE_MODE,
                 root_save_dir: str = conf.SAVE_DIR,
                 save_dir_name: str = conf.SEGMENTATION_SAVE_DIR,
                 color: tuple[int, int, int] = conf.SEGMENTATION_COLOR,
                 weights: str = conf.SEGMENTATION_WEIGHTS_PATH):
        self.model = YOLO(weights)
        self.save = save_mode
        self.color = color

        if save_mode:
            self.save_dir = os.path.join(root_save_dir, save_dir_name)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        self.image = None
        self.image_path = ""
        self.masks = None
    
    def get_image(self) -> np.ndarray | None:
        return self.image
    
    def get_masks(self):
        return self.masks
    
    def predict_image(self, image_path: str) -> int:
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        image_resized = cv2.resize(self.image, (640, 640))
        results = self.model(image_resized)[0]
        
        if results.masks is None:
            if conf.DEBUG_OUTPUT:
                print(f"Для изображения {image_path} не найдено масок.")
            return 0
        
        self.masks = results.masks.data.cpu().numpy()
        if conf.DEBUG_OUTPUT:
            print(f"Обнаружено масок: {len(self.masks)} для изображения {image_path}")
        return len(self.masks)
    
    def calc_masks(self) -> np.ndarray | None:
        if self.masks is None or self.image is None:
            return None

        if self.save:
            image_orig = self.image.copy()

        h_or, w_or = self.image.shape[:2]
        res_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
        for i, mask in enumerate(self.masks):
            mask_resized = cv2.resize(mask, (w_or, h_or))

            mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
            mask[mask_resized > 0] = (255, 255, 255)
            res_mask |= mask
            
            if self.save:
                color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
                color_mask[mask_resized > 0] = self.color

                mask_filename = os.path.join(self.save_dir, f"{os.path.splitext(os.path.basename(self.image_path))[0]}_mask_{i}.png")
                cv2.imwrite(mask_filename, color_mask)

                if conf.DEBUG_OUTPUT:
                    print(f"Сохранена маска: {mask_filename}")

                image_orig = cv2.addWeighted(image_orig, 1.0, color_mask, 0.5, 0)

        if self.save:
            new_image_path = os.path.join(self.save_dir, os.path.splitext(os.path.basename(self.image_path))[0] + '_segmented' + os.path.splitext(self.image_path)[1])
            cv2.imwrite(new_image_path, image_orig)
            
            if conf.DEBUG_OUTPUT:
                print(f"Segmented image saved to {new_image_path}")

        return res_mask

    def train_model(self, data_path: str, epochs: int = 30, imgsz: int = 640, project: str = conf.SEGMENTATION_PROJECT, name: str = 'train_custom') -> None:
        self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            project=project,
            name=name,
            task='segment'
        )
        if conf.DEBUG_OUTPUT:
            print(f"Обучение завершено. Результаты сохранены в {os.path.join(project, name)}")

    def predict(self, source_path: str, save: bool = True, imgsz: int = 640) -> list:
        results = self.model.predict(
            source=source_path,
            save=save,
            imgsz=imgsz,
            show=False,
            hide_labels=True,
            hide_conf=True,
            line_thickness=0,
            task='segment'
        )
        if conf.DEBUG_OUTPUT:
            print(f"Предсказание завершено для: {source_path}")
        return results

