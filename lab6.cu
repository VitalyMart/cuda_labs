#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Количество строк
#define K 1024  // Количество столбцов
#define BLOCK_SIZE 32  // Размер блока для разделяемой памяти

// **1. Ядро без разделяемой памяти**
__global__ void transposeGlobalMemory(float *d_in, float *d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        d_out[x * height + y] = d_in[y * width + x];
    }
}

// **2. Ядро с разделяемой памятью (без разрешения конфликта банков)**
__global__ void transposeSharedMemory(float *d_in, float *d_out, int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE]; // Разделяемая память

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = d_in[y * width + x];
    }
    
    __syncthreads();

    // Транспонируем и записываем в глобальную память
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x < height && y < width) {
        d_out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// **3. Ядро с разделяемой памятью (с разрешением конфликта банков)**
__global__ void transposeSharedMemoryNoBankConflicts(float *d_in, float *d_out, int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // Смещение для предотвращения конфликтов

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = d_in[y * width + x];
    }
    
    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x < height && y < width) {
        d_out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// **Функция для измерения времени выполнения ядра**
float measureKernelExecutionTime(void (*kernel)(float*, float*, int, int), float *d_in, float *d_out, dim3 gridSize, dim3 blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_in, d_out, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}

int main() {
    size_t size = N * K * sizeof(float);
    float *h_in = new float[N * K];
    float *h_out = new float[N * K];

    // Инициализация данных
    for (int i = 0; i < N * K; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // Выделение памяти на GPU
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Копирование данных на GPU
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Определение конфигурации сетки и блоков
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Measuring kernel execution times...\n";

    // Измерение времени выполнения каждого ядра
    float timeGlobal = measureKernelExecutionTime(transposeGlobalMemory, d_in, d_out, gridSize, blockSize);
    float timeShared = measureKernelExecutionTime(transposeSharedMemory, d_in, d_out, gridSize, blockSize);
    float timeNoConflict = measureKernelExecutionTime(transposeSharedMemoryNoBankConflicts, d_in, d_out, gridSize, blockSize);

    std::cout << "Global Memory: " << timeGlobal << " ms\n";
    std::cout << "Shared Memory (no conflict avoidance): " << timeShared << " ms\n";
    std::cout << "Shared Memory (conflict avoidance): " << timeNoConflict << " ms\n";

    // Копирование результата обратно на CPU (необязательно)
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Очистка памяти
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
