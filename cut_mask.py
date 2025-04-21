import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


norm = (lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr)))

def work_image(origin):
    image = np.array(origin) / 255

    gray = np.mean(image, axis=-1)

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
    
    gray_img = (gray_mask * 255).astype(np.uint8)
    new_img = (mask * 255).astype(np.uint8)
    
    return new_img, gray_img

def remove_bg(main_dir, image_name, save_dir):
    img = Image.open(os.path.join(main_dir, "clean", image_name))
    mask = Image.open(os.path.join(main_dir, "mask", image_name.replace(".jpg", "_mask_0.png")))

    img = np.array(img, dtype=np.uint8)
    mask = np.array(mask).max(axis=-1) / 255
    
    white_pixels = np.argwhere(mask == 1.0)
    y_min, x_min = white_pixels.min(axis=0)
    y_max, x_max = white_pixels.max(axis=0)

    mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    img = img[y_min:y_max + 1, x_min:x_max + 1]

    bool_mask = mask == 0

    img[bool_mask, :] = 0
    
    colors = img.reshape(-1, img.shape[-1])
    bright_mask = (colors / 255).mean(axis=-1) >= 0.5
    colors = colors[bright_mask]
    
    colors_flat = colors[:, 0].astype(np.uint32) << 16 | colors[:, 1].astype(np.uint32) << 8 | colors[:, 2].astype(np.uint32)
    
    counts = np.bincount(colors_flat)
    idx = np.argmax(counts)
    
    most_common_color = np.array([(idx >> 16) & 0xFF, (idx >> 8) & 0xFF, idx & 0xFF])
    
    img[bool_mask, 0] = most_common_color[0]
    img[bool_mask, 1] = most_common_color[1]
    img[bool_mask, 2] = most_common_color[2]
    
    _, img = work_image(img)

    image = Image.fromarray(img, mode="L")
    image.save(os.path.join(save_dir, image_name.replace(".jpg", ".png")))


if __name__ == "__main__":
    n = len(os.listdir("formask/clean"))
    for i, image_name in enumerate(os.listdir("formask/clean")):
        print(f"Working image {i + 1}/{n}...")
        remove_bg("formask", image_name, "bg_removed")
