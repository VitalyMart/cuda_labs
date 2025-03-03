#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE (1024 * 1024 * 10)  // 10 миллионов элементов
#define MAX_BLOCK_SIZE 1024
#define ITERATIONS 10

// Ядро для сложения векторов
__global__ void vector_add_kernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// Ядро для скалярного произведения с редукцией
__global__ void dot_product_kernel(float* A, float* B, float* result, int N) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0;
    while (tid < N) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    __syncthreads();
    
    // Редукция
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0)
        atomicAdd(result, cache[0]);
}

// Тестирование скорости копирования
void benchmark_copy(size_t size, bool pinned) {
    float *h_data, *d_data;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    if (pinned) cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
    else h_data = new float[size/4];
    
    cudaMalloc(&d_data, size);
    
    // Host -> Device
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "H2D " << (pinned ? "pinned" : "normal") 
              << " time: " << ms << " ms" << std::endl;
    
    // Device -> Host
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "D2H " << (pinned ? "pinned" : "normal") 
              << " time: " << ms << " ms" << std::endl;

    if (pinned) cudaFreeHost(h_data);
    else delete[] h_data;
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Тестирование сложения векторов с разными блоками
void benchmark_vector_add() {
    const int N = 1024 * 1024 * 10;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    cudaHostAlloc(&h_A, N*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_B, N*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_C, N*sizeof(float), cudaHostAllocDefault);
    
    cudaMalloc(&d_A, N*sizeof(float));
    cudaMalloc(&d_B, N*sizeof(float));
    cudaMalloc(&d_C, N*sizeof(float));
    
    for (int block_size = 32; block_size <= MAX_BLOCK_SIZE; block_size *= 2) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float total_time = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            int grid_size = (N + block_size - 1) / block_size;
            vector_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            total_time += ms;
        }
        std::cout << "Block size " << block_size 
                  << " average time: " << total_time/ITERATIONS << " ms" << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Тестирование скалярного произведения с разными блоками
void benchmark_dot_product() {
    const int N = 1024 * 1024 * 10;
    float *h_A, *h_B;
    float *d_A, *d_B, *d_result;
    
    cudaHostAlloc(&h_A, N*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_B, N*sizeof(float), cudaHostAllocDefault);
    
    cudaMalloc(&d_A, N*sizeof(float));
    cudaMalloc(&d_B, N*sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    for (int block_size = 32; block_size <= 256; block_size *= 2) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float total_time = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_result, 0, sizeof(float));
            
            cudaEventRecord(start);
            int grid_size = (N + block_size - 1) / block_size;
            dot_product_kernel<<<grid_size, block_size>>>(d_A, d_B, d_result, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            total_time += ms;
        }
        std::cout << "Block size " << block_size 
                  << " average time: " << total_time/ITERATIONS << " ms" << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}

int main() {
    std::cout << "=== Memory Copy Benchmark ===" << std::endl;
    std::cout << "Testing normal memory:" << std::endl;
    benchmark_copy(SIZE * sizeof(float), false);
    
    std::cout << "\nTesting pinned memory:" << std::endl;
    benchmark_copy(SIZE * sizeof(float), true);
    
    std::cout << "\n=== Vector Addition Optimization ===" << std::endl;
    benchmark_vector_add();
    
    std::cout << "\n=== Dot Product Optimization ===" << std::endl;
    benchmark_dot_product();
    
    return 0;
}