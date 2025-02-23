#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>

#define BLOCK_SIZE 16

void readMatrix(const std::string &filename, std::vector<float> &A, std::vector<float> &B, int &N) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file!\n";
        exit(EXIT_FAILURE);
    }
    file >> N;
    A.resize(N * N);
    B.resize(N * N);
    for (int i = 0; i < N * N; i++) file >> A[i];
    for (int i = 0; i < N * N; i++) file >> B[i];
}

// CUDA Kernel
__global__ void matMulCUDA(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void multiplyWithRuntimeAPI(const std::vector<float> &A, const std::vector<float> &B, int N) {
    float *d_A, *d_B, *d_C;
    std::vector<float> C(N * N);

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    matMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "CUDA Runtime API time: " 
              << std::chrono::duration<float, std::milli>(end - start).count() 
              << " ms\n";
}

// CUDA Driver API
void multiplyWithDriverAPI(const std::vector<float> &A, const std::vector<float> &B, int N) {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    CUdeviceptr d_A, d_B, d_C;
    std::vector<float> C(N * N);

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    cuMemAlloc(&d_A, N * N * sizeof(float));
    cuMemAlloc(&d_B, N * N * sizeof(float));
    cuMemAlloc(&d_C, N * N * sizeof(float));

    cuMemcpyHtoD(d_A, A.data(), N * N * sizeof(float));
    cuMemcpyHtoD(d_B, B.data(), N * N * sizeof(float));

    cuModuleLoad(&cuModule, "matrix_kernel.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "matMulCUDA");

    void *args[] = { &d_A, &d_B, &d_C, &N };
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    cuLaunchKernel(cuFunction, blocksPerGrid.x, blocksPerGrid.y, 1, 
                   threadsPerBlock.x, threadsPerBlock.y, 1, 
                   0, NULL, args, NULL);
    cuCtxSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cuMemcpyDtoH(C.data(), d_C, N * N * sizeof(float));

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuCtxDestroy(cuContext);

    std::cout << "CUDA Driver API time: " 
              << std::chrono::duration<float, std::milli>(end - start).count() 
              << " ms\n";
}

int main(int argc, char *argv[]) {
    std::vector<float> A, B;
    int N;
    readMatrix("input.txt", A, B, N);

    if (argc > 1 && std::strcmp(argv[1], "--driver") == 0) {
        std::cout << "Running CUDA Driver API...\n";
        multiplyWithDriverAPI(A, B, N);
    } else {
        std::cout << "Running CUDA Runtime API...\n";
        multiplyWithRuntimeAPI(A, B, N);
    }

    return 0;
}
//nvcc lab91.cu -o matrix_cuda -lcuda

//./matrix_cuda --driver
//./matrix_cuda --runner
