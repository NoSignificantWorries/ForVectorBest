import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.filter.filters import ImageProcessor


image_processor = ImageProcessor()
mask_processor = ImageProcessor()

images_path = os.path.join("data", "stickers", "train", "Bad")

for image_path in (glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png"))):
    image_name = os.path.split(image_path)[-1]

    base_name = os.path.splitext(image_name)[0]
    '''
    mask_pattern = os.path.join("tmp", "maski", f"{base_name}*.png")
    mask_files = glob.glob(mask_pattern)
    '''

    # if mask_files:
    image_processor.open(image_path)
        # mask_processor.open(mask_files[0])
    # else:
        # continue
    
    # mask = mask_processor.gray_image()
    # mask_to_save, img = image_processor.cut_mask(mask_processor.get_gray())
    img = image_processor.get_image()
    
    img = 255 - img
    gray = img.mean(axis=-1)
    gray -= gray.min()
    gray /= gray.max()

    img_to_save = (gray * 255).astype(np.uint8)
    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"tmp/neg_gray/{image_name}", img_to_save)
    '''
    mask_to_save = (mask_to_save * 255).astype(np.uint8)
    cv2.imwrite(f"tmp/cut/{base_name}_mask.png", mask_to_save)
    cv2.imwrite(f"tmp/gray/{base_name}.png", gray)
    '''
    
    
    '''
    image_processor.gray_image()
    mask_to_save = image_processor.elements_selection()
    mask_to_save = image_processor.apply_elements_mask()
    
    mask_to_save = (mask_to_save * 255).astype(np.uint8)
    cv2.imwrite(f"tmp/gray/{image_name}", mask_to_save)
    '''
