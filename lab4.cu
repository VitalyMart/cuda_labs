#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 20)  // Размер вектора (2^20 элементов)

__global__ void vector_add_gpu(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Индекс потока
    if (i < n) {
        C[i] = A[i] + B[i];  // Сложение элементов
    }
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Выделение памяти для векторов
    std::vector<float> h_A(N), h_B(N), h_C(N);
    float *d_A, *d_B, *d_C;

    // Заполнение хостовых векторов случайными числами
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));
    check_cuda_error("Memory allocation on device");

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error("Memory copy from host to device");

    // Разные конфигурации блоков и нитей
    int block_sizes[] = {1, 16, 32, 64, 128, 256, 512, 1024};

    for (int block_size : block_sizes) {
        // Вычисляем количество блоков в сетке
        int num_blocks = (N + block_size - 1) / block_size;

        // События для замера времени
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Запуск события
        cudaEventRecord(start);

        // Запуск ядра с разными конфигурациями
        vector_add_gpu<<<num_blocks, block_size>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        check_cuda_error("Kernel launch failed");

        // Запуск события остановки
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Измерение времени
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        std::cout << "Block size: " << block_size << " - Time: " << elapsed_time << " ms" << std::endl;

        // Очистка ресурсов
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Копирование результатов с устройства на хост
    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error("Memory copy from device to host");

    // Проверка правильности результатов (необязательно, для тестирования)
    for (int i = 0; i < N; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-6) {
            std::cerr << "Error at index " << i << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Освобождение памяти на устройстве
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Success!" << std::endl;
    return 0;
}
