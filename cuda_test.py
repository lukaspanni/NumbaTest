import math
import numpy as np

from numba import cuda


@cuda.jit()
def square_root(arr):
    x = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(x, arr.shape[0], stride):
        arr[i] = math.atan(arr[i])


def test_square_root_cuda():
    elements = 10_000
    arr = np.arange(elements, dtype=np.float32)
    print(arr)
    threads_per_block = 128
    block_count = 4
    dev_arr = cuda.to_device(arr)
    square_root[block_count, threads_per_block](dev_arr)
    arr = dev_arr.copy_to_host()
    print(arr)


@cuda.jit()
def kernel_2d(data):
    x, y = cuda.grid(2)
    data[x][y] = x * 10 + y


@cuda.jit()
def kernel_2d_striding(data):
    x, y = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
    for i in range(x, data.shape[0], d1):
        for j in range(y, data.shape[1], d2):
            data[i][j] = (x * 10 + y) * 10 + j


def test_kernel_2d():
    arr = np.zeros(12, dtype=np.float32).reshape(3, 4)
    print(arr)
    print("===================")
    blocks = (1, 2)  # 1 line with 2 columns
    threads = (3, 2)  # 3 lines with 2 columns
    dev_arr = cuda.to_device(arr)
    kernel_2d[blocks, threads](dev_arr)
    arr = dev_arr.copy_to_host()
    print(arr)


def test_kernel_2d_striding():
    arr = np.zeros(16, dtype=np.float32).reshape(4, 4)
    print(arr)
    print("===================")
    blocks = (1,)
    threads = (3, 2)
    dev_arr = cuda.to_device(arr)
    kernel_2d_striding[blocks, threads](dev_arr)
    arr = dev_arr.copy_to_host()
    print(arr)


@cuda.jit()
def easy_matrix_multiply(a, b, c):
    x, y = cuda.grid(2)
    tmp_sum = 0
    for i in range(b.shape[0]):
        tmp_sum += a[x][i] * b[i][y]
    c[x, y] = tmp_sum


@cuda.jit()
def striding_matrix_multiply(a, b, c):
    x, y = cuda.grid(2)
    d1, d2 = cuda.gridsize(2)
    for i in range(x, a.shape[0], d1):
        for j in range(y, b.shape[1], d2):
            tmp_sum = 0
            for k in range(b.shape[0]):
                tmp_sum += a[i][k]*b[k][j]
            c[i,j] = tmp_sum


def test_matrix_multiply(striding=False):
    a = np.arange(10).reshape(5, 2)
    b = np.arange(6).reshape(2, 3)
    c = np.zeros((a.shape[0], b.shape[1]))
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)
    dev_c = cuda.to_device(c)
    print("A:\n", a, "\nB:\n", b)
    print("====================")
    if striding:
        striding_matrix_multiply[(1,), (2, 2)](dev_a, dev_b, dev_c)
    else:
        easy_matrix_multiply[(1,), (5, 3)](dev_a, dev_b, dev_c)
    c = dev_c.copy_to_host()
    print(c)


if __name__ == "__main__":
    test_matrix_multiply(True)
