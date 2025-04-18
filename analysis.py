import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


norm = (lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr)))
get_bins = (lambda x: np.int32(np.ceil(np.log2(x) + 1)))


idx = 33
path = "valid/Good"


def work_image(image):
    global idx, path
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

    mask = gray.copy()
    mask = np.where(mask >= value_range.min(), 0, 1)
    
    gray_mask = gray.copy()
    gray_mask[mask > 0] = 1 - gray_mask[mask > 0]
    gray_mask *= mask
    gray_mask = norm(gray_mask)
    gray_mask[gray_mask < 0.5] = 0.0
    
    gray_img = (gray_mask * 255).astype(np.uint8)
    new_img = (mask * 255).astype(np.uint8)

    new_img = Image.fromarray(new_img, mode="L")
    gray_img = Image.fromarray(gray_img, mode="L")
    
    # new_img.save(f"gray_stickers/{path}/image_{idx}.png")
    gray_img.save(f"gray_stickers_grad/{path}/image_{idx}.png")
    
    _, axises = plt.subplots(nrows=3, ncols=1, figsize=(24, 12))

    # axises[0].plot(val, cnt)
    # axises[0].plot(val, cnt_copy)

    axises[0].hist(gray_mask.flatten(), bins=100, edgecolor='black', linewidth=1, alpha=0.9)
    axises[1].imshow(gray_mask)
    axises[2].imshow(mask)

    print("Process done.")
    # plt.savefig(f"result/Bad/image_{idx}.png", format="png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    read_path = f"stickers/{path}"
    for image_name in os.listdir(read_path):
        origin = Image.open(f"{read_path}/{image_name}")
        work_image(origin)
