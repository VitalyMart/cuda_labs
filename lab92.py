import numpy as np
import time
from numba import cuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Функция для чтения матриц из файла
def read_matrices(filename):
    with open(filename, 'r') as f:
        data = f.read().split()
    N = int(data[0])
    matrix_A = np.array(data[1:N*N+1], dtype=np.float32).reshape(N, N)
    matrix_B = np.array(data[N*N+1:], dtype=np.float32).reshape(N, N)
    return N, matrix_A, matrix_B

# NUMBA CUDA
@cuda.jit
def matmul_numba(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        sum = 0
        for k in range(A.shape[1]):
            sum += A[row, k] * B[k, col]
        C[row, col] = sum

def multiply_with_numba(A, B, N):
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((N, N), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid = (N // 16, N // 16)

    start = time.time()
    matmul_numba[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    end = time.time()

    print(f"Numba CUDA time: {(end - start) * 1000:.2f} ms")
    return d_C.copy_to_host()

# PyCUDA
mod = SourceModule("""
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
""")
matmul_pycuda = mod.get_function("matmul")

def multiply_with_pycuda(A, B, N):
    C = np.zeros((N, N), dtype=np.float32)
    start = time.time()
    matmul_pycuda(
        drv.In(A), drv.In(B), drv.Out(C), np.int32(N),
        block=(16, 16, 1), grid=(N // 16, N // 16)
    )
    end = time.time()

    print(f"PyCUDA time: {(end - start) * 1000:.2f} ms")
    return C

# Главная функция
if __name__ == "__main__":
    N, A, B = read_matrices("input.txt")

    method = input("Выберите метод (numba/pycuda): ").strip().lower()

    if method == "numba":
        print("Running Numba CUDA...")
        C_numba = multiply_with_numba(A, B, N)
        print("Result (Numba):")
        print(C_numba)
    elif method == "pycuda":
        print("\nRunning PyCUDA...")
        C_pycuda = multiply_with_pycuda(A, B, N)
        print("Result (PyCUDA):")
        print(C_pycuda)
    else:
        print("Неверный метод! Введите 'numba' или 'pycuda'.")
