#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Функция для сложения векторов на CPU
void vector_add_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// CUDA ядро для сложения векторов
__global__ void vector_add_gpu(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 20;  // Пример длины вектора (1 миллион)
    size_t size = N * sizeof(float);

    // Выделение памяти для векторов
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C_cpu = (float*)malloc(size);
    float *C_gpu = (float*)malloc(size);

    // Инициализация векторов
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    // 1. Сложение на CPU
    clock_t start = clock();
    vector_add_cpu(A, B, C_cpu, N);
    clock_t end = clock();
    double cpu_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;

    // 2. Сложение на GPU для различных значений потоков
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    std::cout << "Threads per block | Time (seconds)" << std::endl;

    for (int threadsPerBlock = 2; threadsPerBlock <= 1024; threadsPerBlock *= 2) {
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        start = clock();
        vector_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        end = clock();

        double gpu_time = double(end - start) / CLOCKS_PER_SEC;
        std::cout << threadsPerBlock << " | " << gpu_time/1000 << " seconds" << std::endl;
    }

    // 3. Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}
