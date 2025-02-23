#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846
#define THETA_RESOLUTION 360
#define PHI_RESOLUTION 180

// Константная память
__constant__ int c_theta_resolution;
__constant__ int c_phi_resolution;

// Функция на сфере (для GPU)
__device__ float func_on_sphere(float theta, float phi) {
    return sinf(theta) * cosf(phi);
}

// Функция на сфере (для CPU)
float func_on_sphere_cpu(float theta, float phi) {
    return sinf(theta) * cosf(phi);
}

// Заполнение массива значениями функции
void fill_function_data(float *h_data) {
    for (int i = 0; i < THETA_RESOLUTION; ++i) {
        for (int j = 0; j < PHI_RESOLUTION; ++j) {
            float theta = (i * PI) / (THETA_RESOLUTION - 1);
            float phi = (j * 2 * PI) / (PHI_RESOLUTION - 1);
            h_data[i * PHI_RESOLUTION + j] = func_on_sphere_cpu(theta, phi);
        }
    }
}

// Кернел для интеграции с текстурной памятью
__global__ void calculate_integral_texture(cudaTextureObject_t tex, float *result) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < c_theta_resolution && y < c_phi_resolution) {
        float theta = (x * PI) / (c_theta_resolution - 1);
        float phi = (y * 2 * PI) / (c_phi_resolution - 1);

        float dtheta = PI / (c_theta_resolution - 1);
        float dphi = (2 * PI) / (c_phi_resolution - 1);

        float value = tex2D<float>(tex, x, y);
        value *= sinf(theta) * dtheta * dphi;

        atomicAdd(result, value);
    }
}

// Кернел для интеграции без текстурной памяти (ступенчатая интерполяция)
__global__ void calculate_integral_no_texture_step(float *result, float *data) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < c_theta_resolution && y < c_phi_resolution) {
        int idx = x * c_phi_resolution + y;
        
        float theta = (x * PI) / (c_theta_resolution - 1);
        float phi = (y * 2 * PI) / (c_phi_resolution - 1);

        float dtheta = PI / (c_theta_resolution - 1);
        float dphi = (2 * PI) / (c_phi_resolution - 1);

        float value = data[idx] * sinf(theta) * dtheta * dphi;

        atomicAdd(result, value);
    }
}

// Кернел для интеграции без текстурной памяти (линейная интерполяция)
__global__ void calculate_integral_no_texture_linear(float *result, float *data) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < c_theta_resolution - 1 && y < c_phi_resolution - 1) {
        int idx1 = x * c_phi_resolution + y;
        int idx2 = (x + 1) * c_phi_resolution + y;
        int idx3 = x * c_phi_resolution + (y + 1);
        int idx4 = (x + 1) * c_phi_resolution + (y + 1);

        float fx1 = data[idx1];
        float fx2 = data[idx2];
        float fy1 = data[idx3];
        float fy2 = data[idx4];

        float theta = (x * PI) / (c_theta_resolution - 1);
        float phi = (y * 2 * PI) / (c_phi_resolution - 1);

        float dtheta = PI / (c_theta_resolution - 1);
        float dphi = (2 * PI) / (c_phi_resolution - 1);

        float alpha = (x + 0.5f) / float(c_theta_resolution - 1);
        float beta = (y + 0.5f) / float(c_phi_resolution - 1);

        float interpolated_value = (1 - alpha) * (1 - beta) * fx1 +
                                   alpha * (1 - beta) * fx2 +
                                   (1 - alpha) * beta * fy1 +
                                   alpha * beta * fy2;

        interpolated_value *= sinf(theta) * dtheta * dphi;

        atomicAdd(result, interpolated_value);
    }
}

int main() {
    float *d_result;
    float h_result = 0.0f;
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    int h_theta_resolution = THETA_RESOLUTION;
    int h_phi_resolution = PHI_RESOLUTION;
    cudaMemcpyToSymbol(c_theta_resolution, &h_theta_resolution, sizeof(int));
    cudaMemcpyToSymbol(c_phi_resolution, &h_phi_resolution, sizeof(int));

    float *h_data = new float[THETA_RESOLUTION * PHI_RESOLUTION];
    fill_function_data(h_data);

    float *d_data;
    cudaMalloc((void**)&d_data, THETA_RESOLUTION * PHI_RESOLUTION * sizeof(float));
    cudaMemcpy(d_data, h_data, THETA_RESOLUTION * PHI_RESOLUTION * sizeof(float), cudaMemcpyHostToDevice);

    cudaTextureObject_t tex;
    cudaArray *d_array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&d_array, &channelDesc, THETA_RESOLUTION, PHI_RESOLUTION);
    cudaMemcpyToArray(d_array, 0, 0, h_data, THETA_RESOLUTION * PHI_RESOLUTION * sizeof(float), cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((THETA_RESOLUTION + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (PHI_RESOLUTION + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    calculate_integral_texture<<<numBlocks, threadsPerBlock>>>(tex, d_result);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time with texture memory: " << elapsedTime << " ms" << std::endl;

    cudaMemset(d_result, 0, sizeof(float));

    cudaEventRecord(start, 0);
    calculate_integral_no_texture_step<<<numBlocks, threadsPerBlock>>>(d_result, d_data);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time without texture memory (step interpolation): " << elapsedTime << " ms" << std::endl;

    cudaMemset(d_result, 0, sizeof(float));

    cudaEventRecord(start, 0);
    calculate_integral_no_texture_linear<<<numBlocks, threadsPerBlock>>>(d_result, d_data);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time without texture memory (linear interpolation): " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Calculated integral: " << h_result << std::endl;

    cudaDestroyTextureObject(tex);
    cudaFreeArray(d_array);
    cudaFree(d_data);
    cudaFree(d_result);
    delete[] h_data;

    return 0;
}
