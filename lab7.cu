#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846

// Размерность сетки в градусах
#define THETA_RESOLUTION 360
#define PHI_RESOLUTION 180

// Примерная функция, заданная на сфере
__device__ float func_on_sphere(float theta, float phi) {
    // Простая функция, например: f(theta, phi) = sin(theta) * cos(phi)
    return sinf(theta) * cosf(phi);
}

// Кернел для вычисления интеграла с использованием текстурной памяти
__global__ void calculate_integral_texture(float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < THETA_RESOLUTION && idy < PHI_RESOLUTION) {
        float theta = (idx * PI) / (THETA_RESOLUTION - 1);
        float phi = (idy * 2 * PI) / (PHI_RESOLUTION - 1);

        // Вычисление значения функции с учетом веса
        float value = func_on_sphere(theta, phi) * sinf(theta);

        // Атомарное добавление для получения интеграла
        atomicAdd(result, value);
    }
}

// Кернел для вычисления интеграла без текстурной памяти (с программной интерполяцией)
__global__ void calculate_integral_no_texture(float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < THETA_RESOLUTION && idy < PHI_RESOLUTION) {
        float theta = (idx * PI) / (THETA_RESOLUTION - 1);
        float phi = (idy * 2 * PI) / (PHI_RESOLUTION - 1);

        // Программная интерполяция (линейная)
        float value = func_on_sphere(theta, phi) * sinf(theta);

        // Атомарное добавление для получения интеграла
        atomicAdd(result, value);
    }
}

int main() {
    float *d_result;
    float *h_result = (float *)malloc(sizeof(float));

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Инициализация результата
    cudaMemset(d_result, 0, sizeof(float));

    // Устанавливаем размер сетки
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((THETA_RESOLUTION + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (PHI_RESOLUTION + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Сравнение времени для метода с текстурной памятью
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    calculate_integral_texture<<<numBlocks, threadsPerBlock>>>(d_result);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time with texture memory: " << elapsedTime << " ms" << std::endl;

    // Сравнение времени для метода без текстурной памяти
    cudaEventRecord(start, 0);
    calculate_integral_no_texture<<<numBlocks, threadsPerBlock>>>(d_result);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time without texture memory: " << elapsedTime << " ms" << std::endl;

    // Копирование результата с устройства на хост
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Calculated integral: " << *h_result << std::endl;

    // Освобождение памяти
    cudaFree(d_result);
    free(h_result);

    return 0;
}
