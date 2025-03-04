#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

// Макросы для проверки ошибок CUDA и cuBLAS
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCudaError(T result, const char *const func, const char *const file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\"\n", file, line, (int)result, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS_ERROR(val) checkCublasError((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCublasError(T result, const char *const func, const char *const file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d code=%d \"%s\"\n", file, line, (int)result, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Заполнение матрицы случайными значениями (float)
void fillMatrix(float *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (float)rand() / RAND_MAX; // Случайное значение от 0 до 1
    }
}

// Ядро для преобразования float в half (на устройстве)
__global__ void floatToHalfKernel(__half *out, float *in, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        out[idx] = __float2half_rn(in[idx]);
    }
}

// Функция для конвертации матрицы из float в half на устройстве
void convertFloatToHalf(__half *d_half, float *d_float, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    floatToHalfKernel<<<gridSize, blockSize>>>(d_half, d_float, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    srand(time(NULL)); // Инициализация генератора случайных чисел

    // Выделение памяти на хосте для матриц A, B, C (float)
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    fillMatrix(h_A, M*K);
    fillMatrix(h_B, K*N);

    // Выделение памяти на устройстве для матриц (float)
    float *d_A_float, *d_B_float, *d_C_float;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A_float, M*K*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B_float, K*N*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C_float, M*N*sizeof(float)));

    // Копирование данных на устройство
    CHECK_CUDA_ERROR(cudaMemcpy(d_A_float, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B_float, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    // Инициализация cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // Параметры умножения матриц
    float alpha = 1.0f;
    float beta = 0.0f;

    // Измерение времени для обычного умножения (float)
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
                                  &alpha, d_A_float, M, d_B_float, K, 
                                  &beta, d_C_float, M));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float cublasTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&cublasTime, start, stop));
    printf("cuBLAS (float) time: %.3f ms\n", cublasTime);

    // Подготовка данных для тензорных ядер (half)
    __half *d_A_half, *d_B_half;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A_half, M*K*sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B_half, K*N*sizeof(__half)));

    // Конвертация данных из float в half на устройстве
    convertFloatToHalf(d_A_half, d_A_float, M*K);
    convertFloatToHalf(d_B_half, d_B_float, K*N);

    // Выделение памяти для результата (float)
    float *d_C_tensor;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C_tensor, M*N*sizeof(float)));

    // Установка режима Tensor Core
    CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Измерение времени для тензорных ядер
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                    M, N, K, 
                                    &alpha, 
                                    d_A_half, CUDA_R_16F, M,
                                    d_B_half, CUDA_R_16F, K,
                                    &beta, 
                                    d_C_tensor, CUDA_R_32F, M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float tensorCoreTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&tensorCoreTime, start, stop));
    printf("Tensor Core time: %.3f ms\n", tensorCoreTime);

    // Освобождение ресурсов
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_A_float));
    CHECK_CUDA_ERROR(cudaFree(d_B_float));
    CHECK_CUDA_ERROR(cudaFree(d_C_float));
    CHECK_CUDA_ERROR(cudaFree(d_A_half));
    CHECK_CUDA_ERROR(cudaFree(d_B_half));
    CHECK_CUDA_ERROR(cudaFree(d_C_tensor));
    free(h_A);
    free(h_B);
    free(h_C);
    cudaDeviceReset();

    return 0;
}
//nvcc -o lab12 lab12.cu -lcublas
//./lab12 1024 1024 1024