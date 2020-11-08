from time import time

import numpy as np
from numba import stencil, njit, prange, cuda
from PIL import Image

# simple 3x3 gaussian kernel
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.uint16)
gaussian_kernel_factor = 1 / 16

# simple 3x3 sobel kernel
sobel_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.int16)
sobel_kernel_factor = 1 / 8


@njit()
def apply_kernel_cpu(img, kernel, kernel_factor):
    for i in prange(1, img.shape[0] - 1):
        for j in prange(1, img.shape[1] - 1):
            value = np.array([0, 0, 0], dtype=np.int16)
            for k in prange(-1, 2):
                for l in prange(-1, 2):
                    value += kernel[k + 1, l + 1] * img[i + k, j + l]
            img[i, j] = np.ceil(kernel_factor * value)
    return img


@cuda.jit(device=True)
def cuda_rgb_kernel_add(kernel_value, px, value_r, value_g, value_b):
    value_r += kernel_value * px[0]
    value_g += kernel_value * px[1]
    value_b += kernel_value * px[2]
    return value_r, value_g, value_b


@cuda.jit()
def apply_kernel_cuda(img, kernel, kernel_factor):
    x, y = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
    for i in prange(x, img.shape[0], d1):
        for j in prange(y, img.shape[1], d2):
            value_r = 0
            value_g = 0
            value_b = 0
            for k in prange(-1, 2):
                for l in prange(-1, 2):
                    value_r, value_g, value_b = cuda_rgb_kernel_add(kernel[k + 1, l + 1], img[i + k, j + l],
                                                                    value_r, value_g, value_b)
            value_r = kernel_factor * value_r
            value_g = kernel_factor * value_g
            value_b = kernel_factor * value_b
            img[i, j, 0] = value_r
            img[i, j, 1] = value_g
            img[i, j, 2] = value_b


def kernels(img, kernel, kernel_factor, gpu=False):
    if not gpu:
        start = time()
        # algorithm needs int16 array!
        img = apply_kernel_cpu(img.astype(np.int16), kernel, kernel_factor).astype(np.uint8)[:, :, :]
        print(time() - start)
    else:
        start = time()
        img = img.astype(np.int16)
        dev_arr = cuda.to_device(img)
        apply_kernel_cuda[(32, 32), (16, 16)](dev_arr, kernel, kernel_factor)
        img = dev_arr.copy_to_host()
        print(time() - start)

    return img


if __name__ == "__main__":
    test_img = Image.open("simple_test_img.jpg")
    data = np.array(test_img)
    mainpulated_img = kernels(data, sobel_kernel, sobel_kernel_factor)
    Image.fromarray(mainpulated_img).show()
    exit()
