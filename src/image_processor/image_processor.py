import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import src.conf as conf
from src.base_worker import BaseWorker


class Preprocessor(BaseWorker):
    def __init__(self):
        self.image = None
        self.gray = None
        self.save_dir = os.path.join(conf.SAVE_DIR, conf.PREPROCESS_SAVE_DIR)

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        if conf.DEBUG_OUTPUT:
            print("Preprocessor called with image shape:", image.shape)

        self.image = image.astype(np.float64)

        self.gray = 255 - self.image
        self.gray[:, :, 0] = self.gray.mean(axis=-1)
        self.gray[:, :, 1] = self.gray[:, :, 0]
        self.gray[:, :, 2] = self.gray[:, :, 0]
        return self.gray.astype(np.uint8)
    
    def visualize(self) -> None:
        if self.gray is None:
            return None
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        to_save = self.gray.astype(np.uint8)
        cv2.imwrite(os.path.join(self.save_dir, "processed_image_0.jpg"), to_save)
