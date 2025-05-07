import cv2
import numpy as np
import matplotlib.pyplot as plt


def rects(img: np.ndarray, size: int):
    img = img.copy()

    for i in range(img.shape[0] // size):
        for j in range(img.shape[1] // size):
            y, x = i * size, j * size
            img[y:y + size, x:x + size] = img[y:y + size, x:x + size].max()
    
    return img


def main(file: str) -> None:
    orig_image = cv2.imread(file)
    img = orig_image.astype(np.float32)
    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
    img /= 255
    
    img = np.mean(img, axis=-1)
    H, W = img.shape
    if H % 2 != 0:
        img = img[:-1, :]
        H -= 1
    if W % 2 != 0:
        img = img[:, :-1]
        W -= 1
    
    paddedH = int(2 ** np.ceil(np.log2(H)))
    padH = (paddedH - H) // 2
    paddedW = int(2 ** np.ceil(np.log2(W)))
    padW = (paddedW - W) // 2
    
    resized_img = np.zeros((paddedH, paddedW), dtype=np.float32)
    resized_img[padH:padH + H, padW:padW + W] = img
    
    r0 = rects(resized_img, 512)
    r1 = rects(resized_img, 256)
    r2 = rects(resized_img, 128)
    r3 = rects(resized_img, 64)
    r4 = rects(resized_img, 32)
    r5 = rects(resized_img, 16)
    
    R = r2 + r3 + r5

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))

    ax[0][0].imshow(img, cmap="inferno")
    ax[0][1].imshow(R, cmap="inferno")

    ax[1][0].imshow(r0, cmap="inferno")
    ax[2][0].imshow(r1, cmap="inferno")
    ax[3][0].imshow(r2, cmap="inferno")

    ax[1][1].imshow(r3, cmap="inferno")
    ax[2][1].imshow(r4, cmap="inferno")
    ax[3][1].imshow(r5, cmap="inferno")
    
    plt.savefig("res.png", format="png", dpi=300)
    

if __name__ == "__main__":
    main("gray_stickers_grad/train/Bad/image_10.png")
