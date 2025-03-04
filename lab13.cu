#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

__global__ void wmma_kernel(half* a, half* b, float* c, int M, int N, int K) {
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int c_row = blockIdx.y * blockDim.y * WMMA_M + warp_id * WMMA_M;
    int c_col = blockIdx.x * blockDim.x * WMMA_N;

    if (c_row >= M || c_col >= N) return;

    for (int i = 0; i < K; i += WMMA_K) {
        int a_col = i;
        int b_row = i;

        wmma::load_matrix_sync(a_frag, a + c_row * K + a_col, K);
        wmma::load_matrix_sync(b_frag, b + b_row * N + c_col, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(c + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
}

void test_cublas(int M, int N, int K) {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    // Allocate host memory
    h_A = new float[size_A];
    h_B = new float[size_B];
    h_C = new float[size_C];

    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // Allocate device memory
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N,
                d_A, K,
                &beta, d_C, N);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 10; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N,
                    d_A, K,
                    &beta, d_C, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "cuBLAS time: " << milliseconds / 10 << " ms" << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

void test_wmma(int M, int N, int K) {
    half *h_A, *h_B;
    float *h_C;
    half *d_A, *d_B;
    float *d_C;

    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    // Allocate host memory
    h_A = new half[size_A];
    h_B = new half[size_B];
    h_C = new float[size_C];

    // Initialize with random data
    float* temp = new float[size_A];
    randomInit(temp, size_A);
    for (int i = 0; i < size_A; ++i) h_A[i] = __float2half(temp[i]);
    randomInit(temp, size_B);
    for (int i = 0; i < size_B; ++i) h_B[i] = __float2half(temp[i]);
    delete[] temp;

    // Allocate device memory
    cudaMalloc(&d_A, size_A * sizeof(half));
    cudaMalloc(&d_B, size_B * sizeof(half));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(half), cudaMemcpyHostToDevice);

    // Setup grid and block
    dim3 grid((N + WMMA_N * 4 - 1) / (WMMA_N * 4),
              (M + WMMA_M * 4 - 1) / (WMMA_M * 4));
    dim3 block(128);

    // Warmup
    wmma_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 10; ++i) {
        wmma_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "WMMA time: " << milliseconds / 10 << " ms" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    std::vector<int> sizes = {256, 512, 1024, 2048};

    for (int size : sizes) {
        int M = size, N = size, K = size;
        std::cout << "\nMatrix size: " << size << "x" << size << std::endl;
        
        // Check if size is multiple of 16 for WMMA
        if (size % 16 != 0) {
            std::cout << "Size must be multiple of 16 for WMMA" << std::endl;
            continue;
        }

        test_cublas(M, N, K);
        test_wmma(M, N, K);
    }

    return 0;
}

//nvcc -arch=sm_70 -o lab13 lab13.cu -lcublas