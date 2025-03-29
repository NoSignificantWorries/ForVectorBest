import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def mask_of_dirt(image):
    mask = image[:, :, 0] * image[:, :, 1] * image[:, :, 2]
    mask -= np.min(mask)
    mask /= np.max(mask)
    
    return mask

def get_max_range(elem_, data_, bound):
    idxl = elem_[0] - 1
    last_lower = elem_[1]
    idxu = elem_[0] + 1
    last_upper = elem_[1]
    result = [elem_]
    while idxl >= 0 or idxu < len(data_):
        if idxl >= 0:
            if data_[idxl][1] >= bound and data_[idxl][1] <= last_lower:
                result.append(data_[idxl])
                idxl -= 1
            else:
                idxl = -1
        if idxu < len(data_):
            if data_[idxu][1] >= bound and data[idxu][1] <= last_upper:
                result.append(data_[idxu])
                idxu += 1
            else:
                idxu = len(data_)
    
    result.sort(key=(lambda x: x[4]))
    return result, result[0][2], result[-1][3]


if __name__ == "__main__":
    '''
    path = "Bad"
    images = []
    for image in os.listdir(path):
        mask = mask_of_dirt(Image.open(f"{path}/{image}"))
        plt.imshow(mask, cmap="inferno")
        plt.savefig(f"result/{path}/{image}", format="png", dpi=600)
    '''
    origin = Image.open("Bad/IMG_1323_jpg.rf.b852d857b2b6a164222e5f8d6d8f66f6.jpg")
    origin = np.array(origin) / 255
    mask = mask_of_dirt(origin)
    # mask = mask_of_dirt(Image.open("Bad/IMG_1321_jpg.rf.c2695d7de507aefede575904c0a3f3aa.jpg"))
    
    pixels = mask.flatten()
    
    get_bins = (lambda x: np.int32(np.ceil(np.log2(x) + 1)))

    values, count = np.unique(pixels, return_counts=True)
    n = len(values)
    print(n)

    cnt, val = np.histogram(pixels, bins=get_bins(len(pixels)))
    count_val, count_bins = np.histogram(cnt, bins=get_bins(len(cnt)))
    idx = np.argmax(count_val)
    upper_bound = count_bins[idx + 1]
    
    data = []
    i = 0
    for j, elem in enumerate(cnt):
        data.append([j, elem, val[j], val[j + 1], (val[j] + val[j + 1]) / 2, elem >= upper_bound])
    # print(*data, sep="\n")
    
    maximums = []
    for j in range(len(cnt)):
        if j == 0:
            if data[j][1] > data[j + 1][1] and data[j][-1]: maximums.append(data[j])
        elif j == len(cnt) - 1:
            if data[j][1] > data[j - 1][1] and data[j][-1]: maximums.append(data[j])
        else:
            if data[j][1] > data[j - 1][1] and data[j][1] > data[j + 1][1] and data[j][-1]: maximums.append(data[j])
    print(*maximums, sep="\n")

    data_ranges = []
    avg = 0
    for max_ in maximums:
        res, left, right = get_max_range(max_, data, upper_bound)
        avg += max_[4]
        data_ranges.append([max_, left, right, res])
    avg /= len(maximums)

    # print(data_ranges)
    
    # new_mask = np.full(mask.shape, avg)
    new_mask = np.zeros(mask.shape)
    new_mask[(0 <= mask) & (mask <= data_ranges[0][1])] = 0
    new_mask[(data_ranges[0][1] <= mask) & (mask <= data_ranges[0][2])] = 0.1
    new_mask[(data_ranges[0][2] <= mask) & (mask <= data_ranges[1][1])] = 0.5
    new_mask[(data_ranges[1][1] <= mask) & (mask <= data_ranges[1][2])] = 1
    new_mask[(data_ranges[1][2] <= mask) & (mask <= 1)] = 1

    fig, axises = plt.subplots(nrows=3, ncols=2)

    axises[0][0].plot(values, count)
    axises[1][0].hist(pixels, bins=get_bins(len(pixels)), color='skyblue', edgecolor='black')
    axises[1][0].hlines(count_bins[idx + 1], 0, 1, linestyle="dotted")
    axises[2][0].imshow(origin)
    axises[2][1].imshow(new_mask)
    
    plt.show()
