from time import time

import numpy as np
from numba import stencil, njit, prange, cuda
from PIL import Image


@stencil(neighborhood=((-1, 1), (-1, 1)))
def avg(arr):
    return np.mean(arr[-1:2, -1:2])


@njit(parallel=True)
def avg_filter(arr):
    return avg(arr)


@njit()
def greyscale_avg(px_val):
    return np.mean(px_val)


@njit()
def greyscale_luminosity(px_val):
    return px_val[0] * 0.21 + px_val[1] * 0.72 + px_val[2] * 0.07


@njit()
def greyscale_lightness(px_val):
    return 0.5 * (np.max(px_val) + np.min(px_val))


@njit(parallel=True)
def greyscale(img_data, alg):
    for i in prange(img_data.shape[0]):
        for j in prange(img_data.shape[1]):
            img_data[i, j] = alg(img_data[i, j])
    return img_data


@stencil(neighborhood=((0, 0), (0, 0), (0, 2)))
def greyscale_stencil_avg(px):
    return np.mean(px[0, 0, 0:3])


@stencil(neighborhood=((0, 0), (0, 0), (0, 2)))
def greyscale_stencil_luminosity(px):
    return px[0, 0, 0] * 0.21 + px[0, 0, 1] * 0.72 + px[0, 0, 2] * 0.07


@stencil(neighborhood=((0, 0), (0, 0), (0, 2)))
def greyscale_stencil_lightness(px):
    return 0.5 * (np.max(px[0, 0, 0:3]) + np.min(px[0, 0, 0:3]))


# Multiple functions needed because numba does not like stencils as parameter
@njit(parallel=True)
def greyscale_with_stencil_avg(arr):
    return greyscale_stencil_avg(arr)


@njit(parallel=True)
def greyscale_with_stencil_luminosity(arr):
    return greyscale_stencil_luminosity(arr)


@njit(parallel=True)
def greyscale_with_stencil_lightness(arr):
    return greyscale_stencil_lightness(arr)


@cuda.jit(device=True)
def gpu_avg(px_val):
    return (px_val[0] + px_val[1] + px_val[2]) // 3


@cuda.jit(device=True)
def gpu_luminosity(px_val):
    return (px_val[0] * 0.21 + px_val[1] * 0.72 + px_val[2] * 0.07) // 1


@cuda.jit(device=True)
def gpu_lightness(px_val):
    return (0.5 * (max(px_val[0], px_val[1], px_val[2]) + min(px_val[0], px_val[1], px_val[2]))) // 1


# Multiple functions needed because numba also does not like device funtions as parameter
@cuda.jit()
def gpu_greyscale_avg(img):
    x, y = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
    for i in range(x, img.shape[0], d1):
        for j in range(y, img.shape[1], d2):
            img[i][j] = gpu_avg(img[i][j])


@cuda.jit()
def gpu_greyscale_luminosity(img):
    x, y = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
    for i in range(x, img.shape[0], d1):
        for j in range(y, img.shape[1], d2):
            img[i][j] = gpu_luminosity(img[i][j])


@cuda.jit()
def gpu_greyscale_lightness(img):
    x, y = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
    for i in range(x, img.shape[0], d1):
        for j in range(y, img.shape[1], d2):
            img[i][j] = gpu_lightness(img[i][j])


def crop(img, new_size):
    y, x, _ = img.shape
    start_x = x // 2 - (new_size[1] // 2)
    start_y = y // 2 - (new_size[0] // 2)
    return img[start_y:start_y + new_size[0], start_x:start_x + new_size[1]]


algorithms = {
    "avg": {"cpu": greyscale_avg, "stencil": greyscale_with_stencil_avg, "gpu": gpu_greyscale_avg},
    "luminosity": {"cpu": greyscale_luminosity, "stencil": greyscale_with_stencil_luminosity,
                   "gpu": gpu_greyscale_luminosity},
    "lightness": {"cpu": greyscale_lightness, "stencil": greyscale_with_stencil_lightness,
                  "gpu": gpu_greyscale_lightness}
}


def test_greyscale(img, alg, stencil=False, gpu=False):
    if not stencil and not gpu:
        print(img.shape)
        start = time()
        img = greyscale(img, algorithms[alg]["cpu"])
        print(time() - start)
    elif stencil and not gpu:
        start = time()
        # works!
        img = algorithms[alg]["stencil"](img).astype(np.uint8)[:, :, 0]
        print(time() - start)
    elif not stencil and gpu:
        start = time()
        dev_arr = cuda.to_device(img)
        algorithms[alg]["gpu"][(32, 32), (16, 16)](dev_arr)
        img = dev_arr.copy_to_host()
        print(time() - start)
    else:
        raise ValueError("Either stencil or gpu has to be False")
    return img


def multiple(images):
    img_data_arrays = []
    s_time = time()
    for image in images:
        test_img = Image.open(image)
        data = np.array(test_img)
        test_img = None
        img_data_arrays.append(test_greyscale(data, "avg", gpu=True))
        img_data_arrays.append(test_greyscale(data, "lightness", gpu=True))
        img_data_arrays.append(test_greyscale(data, "luminosity", gpu=True))
        data = None
    print("GPU: Total for 3 Images and 3 greyscales each", time() - s_time)
    s_time = time()
    for image in images:
        test_img = Image.open(image)
        data = np.array(test_img)
        test_img = None
        img_data_arrays.append(test_greyscale(data, "avg", stencil=True))
        img_data_arrays.append(test_greyscale(data, "lightness", stencil=True))
        img_data_arrays.append(test_greyscale(data, "luminosity", stencil=True))
        data = None
    print("Stencil: Total for 3 Images and 3 greyscales each", time() - s_time)


if __name__ == "__main__":
    multiple(["test_img.jpg", "test_img_2.jpg", "test_img_3.jpg"])
    test_img = Image.open("test_img_2.jpg")
    data = np.array(test_img)

    exit()

    img_data = test_greyscale(data.copy(), "avg", gpu=True)
    Image.fromarray(img_data).show()
    exit()

    # crop image
    img_data = data.copy()
    print(img_data.shape)
    crop_size = (3000, 3000)
    print("Crop to: ", crop_size)
    img_data = crop(img_data, crop_size)
    print(img_data.shape)
    Image.fromarray(img_data).show()

    # avg filter
    img_data = avg_filter(img_data[:, :, 0])
