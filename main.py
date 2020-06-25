from numba import jit, njit, int32, prange
import numpy as np
import time


@jit(nopython=True)
def go_fast(a):  # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i, j]
    return result


@njit
def colsum(arr):
    r, c = arr.shape
    result = np.zeros(c)
    for i in range(r):
        for j in range(c):
            result[j] += arr[i, j]
    return result


@njit
def rowsum(arr):
    r, c = arr.shape
    result = np.zeros(r)
    for i in range(r):
        for j in range(c):
            result[i] += arr[i, j]
    return result


@njit(parallel=True)
def test_parallel(x):
    n = x.shape[0]
    a = np.sin(x)
    b = np.cos(a * a)
    acc = 0
    for i in prange(n - 2):
        for j in prange(n - 1):
            acc += b[i] + b[j + 1]
    return acc


# Numba Free
def nf_go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


def nf_sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i, j]
    return result


def nf_colsum(arr):
    r, c = arr.shape
    result = np.zeros(c)
    for i in range(r):
        for j in range(c):
            result[j] += arr[i, j]
    return result


def nf_rowsum(arr):
    r, c = arr.shape
    result = np.zeros(r)
    for i in range(r):
        for j in range(c):
            result[i] += arr[i, j]
    return result


def test(x, y, z):
    sum2d(y)
    colsum(z)
    rowsum(z)
    go_fast(x)


def test_numbafree(x, y, z):
    nf_sum2d(y)
    nf_colsum(z)
    nf_rowsum(z)
    nf_go_fast(x)


test_parallel(np.arange(10))
# test_parallel.parallel_diagnostics(level=4)


x = np.arange(10000).reshape(100, 100)
y = np.array([[pow(2, x) + y for x in range(20)] for y in range(5000)])
z = np.arange(5000).reshape(100, 50)

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
test(x, y, z)
end = time.time()
with_compilation_time = end - start
print("Elapsed (with compilation) = %s" % with_compilation_time)

iterations = 100
# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
for i in range(iterations):
    test(x, y, z)
end = time.time()
numba_time = end - start
print("Elapsed (after compilation) = %s" % numba_time)

start = time.time()
for i in range(iterations):
    test_numbafree(x, y, z)
end = time.time()
python_time = end - start

print("Elapsed (without numba) = %s" % python_time)
print("Python time/run: ", (python_time / iterations), "Numba time/run", (numba_time / iterations))
print("Speedup:", python_time / numba_time, "Absolute Speedup: ", python_time - numba_time,
      "Absolute Speedup with compilation ((n+1) runs total):", python_time - (with_compilation_time + numba_time))

