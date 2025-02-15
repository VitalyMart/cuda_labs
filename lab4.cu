#include <iostream>
#include <cuda_runtime.h>

#define VECTOR_SIZE (1 << 20)  // 1048576 элементов

// Ядро CUDA для сложения двух векторов
__global__ void vectorAdd(const float *A, const float *B, float *C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

// Функция для замера времени выполнения с использованием CUDA events
float measureKernelExecutionTime(int threadsPerBlock) {
    float *d_A, *d_B, *d_C;
    float *h_A = new float[VECTOR_SIZE];
    float *h_B = new float[VECTOR_SIZE];
    float *h_C = new float[VECTOR_SIZE];

    // Инициализация входных данных
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(VECTOR_SIZE - i);
    }

    // Выделение памяти на GPU
    cudaMalloc((void**)&d_A, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&d_B, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C, VECTOR_SIZE * sizeof(float));

    // Копирование данных на GPU
    cudaMemcpy(d_A, h_A, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Определение конфигурации сетки и блоков
    int blocksPerGrid = (VECTOR_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Создание CUDA событий
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запуск и замер времени выполнения
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, VECTOR_SIZE);
    cudaEventRecord(stop);

    // Ожидание завершения выполнения ядра
    cudaEventSynchronize(stop);

    // Измерение времени
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копирование результата обратно на CPU
    cudaMemcpy(h_C, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Очистка памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return milliseconds;
}

int main() {
    int threadConfigs[] = {1, 16, 32, 64, 128, 256, 512, 1024};

    std::cout << "Threads per block | Time (ms)" << std::endl;
    std::cout << "-----------------|----------" << std::endl;

    for (int threadsPerBlock : threadConfigs) {
        float time = measureKernelExecutionTime(threadsPerBlock);
        std::cout << threadsPerBlock << "               | " << time << " ms" << std::endl;
    }

    return 0;
}
