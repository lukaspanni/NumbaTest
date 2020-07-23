from time import time

import numpy as np
from numba import stencil, njit, prange
from PIL import Image


@stencil(neighborhood=((-1, 1), (-1, 1)))
def avg(arr):
    return np.mean(arr[-1:2, -1:2])


@njit(parallel=True)
def avg_filter(arr):
    return avg(arr)


@njit(parallel=True)
def greyscale(img_data):
    for i in prange(img_data.shape[0]):
        for j in prange(img_data.shape[1]):
            img_data[i, j] = np.mean(img_data[i, j])
    return img_data


@stencil(neighborhood=((0, 0), (0, 0), (0, 2)))
def greyscale_stencil(px):
    return np.mean(px[0, 0, 0:3])


@njit(parallel=True)
def greyscale_with_stencil(arr):
    return greyscale_stencil(arr)


def crop(img, new_size):
    y, x, _ = img.shape
    start_x = x // 2 - (new_size[1] // 2)
    start_y = y // 2 - (new_size[0] // 2)
    return img[start_y:start_y + new_size[0], start_x:start_x + new_size[1]]


if __name__ == "__main__":
    test_img = Image.open("test_img_2.jpg")
    data = np.array(test_img)
    img_data = data.copy()
    print(img_data.shape)
    start = time()
    img_data = greyscale(img_data)
    print(time() - start)
    Image.fromarray(img_data).show()

    img_data = data.copy()
    print(img_data.shape)
    start = time()
    # does not look nice but works
    img_data = greyscale_with_stencil(img_data).astype(np.uint8)[:, :, 0]
    print(time() - start)
    Image.fromarray(img_data).show()

    crop_size = (3000, 3000)
    print("Crop to: ", crop_size)
    img_data = crop(img_data, crop_size)
    print(img_data.shape)

    start = time()
    img_data = avg_filter(img_data[:, :, 0])
    print(time() - start)

    Image.fromarray(img_data).show()

    exit()
    test_arr = np.arange(500 * 500).reshape(500, 500)
    print(test_arr)
    avg_arr = avg_filter(test_arr)
    print(avg_arr)
