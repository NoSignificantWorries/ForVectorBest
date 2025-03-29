import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import cv2


norm = (lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr)))
get_bins = (lambda x: np.int32(np.ceil(np.log2(x) + 1)))


def mask_of_dirt(image):
    mask = image[:, :, 0] * image[:, :, 1] * image[:, :, 2]
    
    return norm(mask)


origin = Image.open("bg_remove_results/IMG_1326_1_png.rf.1882eae7fcb881ff4747ff88490dd0cb_bg_removed.jpg")
image = np.array(origin) / 255

gray = np.mean(image, axis=2)
ax1, ax2 = gray.shape

scale_factor = 10

scaled_gray = np.zeros((ax1 * scale_factor, ax2 * scale_factor))
for i in range(ax1):
    for j in range(ax2):
        scaled_gray[i * scale_factor:scale_factor * (i + 1), j * scale_factor:scale_factor * (j + 1)] = gray[i, j]

'''
# 1. Увеличение изображения (используем бикубическую интерполяцию, как более качественную)
scale_factor = 10
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

sharpen_amount = 100.0
# 2. Повышение резкости (используем unsharp mask)
blurred = cv2.GaussianBlur(resized_image, (5, 5), 1.0)
sharpened = float(sharpen_amount + 1) * resized_image - float(sharpen_amount) * blurred
sharpened = np.maximum(sharpened, 0)  # Обрезаем значения ниже 0
sharpened = np.minimum(sharpened, 1) # Обрезаем значения выше 255
'''

std_image = scipy.ndimage.generic_filter(scaled_gray, np.std, size=5, mode='nearest')
std_image = norm(std_image)

pixels = std_image.flatten()
grad = np.histogram(pixels, bins=100)
idx = 3
mask = np.where((grad[1][0] <= std_image) & (std_image <= grad[1][idx * 2 + 1]), 0, 1)

_, axises = plt.subplots(nrows=3, ncols=2)

axises[0][0].imshow(scaled_gray)
axises[1][0].imshow(std_image)
axises[2][0].imshow(mask, cmap="inferno")
axises[0][1].hist(pixels, bins=100, color='skyblue', edgecolor='black')

plt.savefig("result_rms.png", format="png", dpi=600)
plt.show()
