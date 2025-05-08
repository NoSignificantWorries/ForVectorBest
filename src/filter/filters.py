import os

import numpy as np
import cv2

# custom modules
import src.conf as conf


def norm(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


class ImageProcessor:
    def __init__(self, save_mode: bool = conf.SAVE_MODE, save_dir: str = conf.SAVE_DIR, filters_save_dir: str = conf.FILTERS_SAVE_DIR):
        self.image = None
        self.gray = None
        self.elements_mask = None
        self.gray_elements = None

        self.save = save_mode

        if self.save:
            self.save_path = os.path.join(save_dir, filters_save_dir)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
    
    def open(self, image_path: str) -> np.ndarray | None:
        if not os.path.exists(image_path):
            return None

        self.image = cv2.imread(image_path)
        return self.image
    
    def gray_image(self) -> np.ndarray | None:
        if self.image is None:
            return None
        
        img = self.image.astype(np.float64) / 255
        self.gray = np.mean(img, axis=-1)

        return self.gray
    
    def get_gray(self) -> np.ndarray | None:
        return self.gray
    
    def get_image(self) -> np.ndarray | None:
        return self.image
    
    def elements_selection(self) -> np.ndarray | None:
        if self.gray is None or self.image is None:
            return None
        
        val, cnt = np.unique(self.gray, return_counts=True)
        std = np.mean(cnt)
        mask_cnt = cnt <= std
        mask_val = val < 0.3
        cnt_copy = cnt.copy()
        cnt_copy[mask_cnt | mask_val] = 0

        value_range = val[cnt_copy > 0]

        mask = self.gray.copy()
        mask = np.where(mask >= value_range.min(), 0, 1)
        
        self.elements_mask = mask

        return self.elements_mask
    
    def apply_elements_mask(self) -> np.ndarray | None: 
        if self.gray is None or self.elements_mask is None:
            return None

        gray_mask = self.gray.copy()
        gray_mask[self.elements_mask > 0] = 1 - gray_mask[self.elements_mask > 0]
        gray_mask *= self.elements_mask
        gray_mask = norm(gray_mask)
        
        self.gray_elements = gray_mask

        return self.gray_elements
    
    def cut_mask(self, mask: np.ndarray) -> np.ndarray | None:
        if self.image is None or mask is None:
            return None
        
        white_pixels = np.argwhere(mask == 1.0)
        y_min, x_min = white_pixels.min(axis=0)
        y_max, x_max = white_pixels.max(axis=0)

        mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        self.image = self.image[y_min:y_max + 1, x_min:x_max + 1]

        bool_mask = mask == 0

        self.image[bool_mask, :] = 0
        
        colors = self.image.reshape(-1, self.image.shape[-1])
        
        colors_flat = colors[:, 0].astype(np.uint32) << 16 | colors[:, 1].astype(np.uint32) << 8 | colors[:, 2].astype(np.uint32)
        
        counts = np.bincount(colors_flat)
        idx = np.argmax(counts)
        
        most_common_color = np.array([(idx >> 16) & 0xFF, (idx >> 8) & 0xFF, idx & 0xFF])
        
        self.image[bool_mask, 0] = most_common_color[0]
        self.image[bool_mask, 1] = most_common_color[1]
        self.image[bool_mask, 2] = most_common_color[2]
        
        return self.image
