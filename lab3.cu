#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// CUDA ядро для сложения векторов с ошибкой обращения к памяти
__global__ void vector_add_gpu(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Намеренная ошибка: если индекс больше, чем N/2, то обращаемся за границы массива
    if (i >= N / 2) {
       // printf("Thread %d is trying to access out of bounds memory!\n", i);  // Добавлен вывод для отладки
        C[i] = A[i] + B[i];  // Ошибка: доступ к памяти за пределами массива
        
    }
}

void checkCudaError(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

int main() {
    int N = 1 << 20;  // Пример длины вектора (1 миллион)
    size_t size = N * sizeof(float);

    float *A, *B, *C_gpu;
    float *d_A, *d_B, *d_C;

    // Выделение памяти на хосте и на устройстве
    cudaMalloc((void**)&d_A, size);
    checkCudaError("cudaMalloc for d_A failed");
    cudaMalloc((void**)&d_B, size);
    checkCudaError("cudaMalloc for d_B failed");
    cudaMalloc((void**)&d_C, size);
    checkCudaError("cudaMalloc for d_C failed");

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C_gpu = (float*)malloc(size);

    // Инициализация векторов
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    // Копирование данных на устройство
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy for d_A failed");
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy for d_B failed");

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск CUDA-ядра
    vector_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError("CUDA kernel launch failed");

    // Проверка на ошибки после синхронизации
    cudaDeviceSynchronize();
    checkCudaError("CUDA error after kernel execution");

    // Копирование результатов на хост
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy for C_gpu failed");

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C_gpu);

    return 0;
}
