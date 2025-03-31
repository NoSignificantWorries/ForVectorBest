import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


norm = (lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr)))
get_bins = (lambda x: np.int32(np.ceil(np.log2(x) + 1)))


def mask_of_dirt(image):
    mask = image[:, :, 0] * image[:, :, 1] * image[:, :, 2]
    
    return norm(mask)


idx = 0


def work_image(image):
    global idx
    idx += 1
    print(f"Image {idx} start...")
    image = np.array(origin) / 255

    gray = np.mean(image, axis=2)

    val, cnt = np.unique(gray, return_counts=True)
    std = np.mean(cnt)
    mask_cnt = cnt <= std
    mask_val = val < 0.3
    cnt_copy = cnt.copy()
    cnt_copy[mask_cnt | mask_val] = 0

    value_range = val[cnt_copy > 0]

    gray_copy = gray.copy()
    gray_copy = np.where(gray_copy >= value_range.min(), 0, 1)

    _, axises = plt.subplots(nrows=2, ncols=1, figsize=(24, 12))

    axises[0].plot(val, cnt)
    axises[0].plot(val, cnt_copy)

    axises[1].imshow(gray_copy)

    print("Process done.")
    plt.savefig(f"result/Bad/image_{idx}.png", format="png", dpi=600)
    plt.show()


if __name__ == "__main__":
    path = "Bad"
    for image_path in os.listdir(path):
        origin = Image.open(f"{path}/{image_path}")
        work_image(origin)        
