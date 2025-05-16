import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


images_path = os.path.join("data", "stickers2", "Bad")

for image_path in (glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png"))):
    image_name = os.path.split(image_path)[-1]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    
    img = 255 - img
    gray = img.mean(axis=-1)
    gray -= gray.min()
    gray /= gray.max()

    img_to_save = (gray * 255).astype(np.uint8)
    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"tmp/neg_gray/{image_name}", img_to_save)
