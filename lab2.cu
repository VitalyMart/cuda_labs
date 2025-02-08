#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add_gpu(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i]; // Сложение двух векторов
    }
}

int main() {
    int N = 1000; // Размерность вектора
    float *A, *B, *C; // Указатели на векторы
    float *d_A, *d_B, *d_C; // Указатели на векторы на устройстве

    size_t size = N * sizeof(float);

    // Выделение памяти для векторов на хосте
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Заполнение векторов A и B случайными значениями
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск ядра на устройстве
    vector_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Ожидание завершения ядра
    cudaDeviceSynchronize();

    // Копирование результата с устройства на хост
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Проверка на ошибку
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
