#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>

// Функция для вычисления скалярного произведения с использованием CUDA
__global__ void dot_product_kernel(float* a, float* b, float* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(result, a[idx] * b[idx]);
    }
}

// Функтор для умножения элементов с использованием шаблона
struct multiply_and_add {
    template<typename Tuple>
    __host__ __device__
    float operator()(Tuple t) const {
        return thrust::get<0>(t) * thrust::get<1>(t);
    }
};

float dot_product_thrust(const std::vector<float>& a, const std::vector<float>& b, int n) {
    // Копируем данные в device_vector через итераторы
    thrust::device_vector<float> d_a(a.begin(), a.end());
    thrust::device_vector<float> d_b(b.begin(), b.end());

    // Применение transform_reduce с функтором multiply_and_add
    return thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(d_a.begin(), d_b.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_a.end(), d_b.end())),
        multiply_and_add(),
        0.0f,
        thrust::plus<float>()
    );
}

// Функция для транспонирования матрицы с использованием CUDA
__global__ void transpose_kernel(float* input, float* output, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        output[idy * width + idx] = input[idx * height + idy];
    }
}

// Функция для транспонирования матрицы с использованием Thrust
void transpose_thrust(const std::vector<float>& input, std::vector<float>& output, int width, int height) {
    thrust::device_vector<float> d_input(input.begin(), input.end());
    thrust::device_vector<float> d_output(output.size());

    // Получаем сырые указатели на данные
    float* d_input_ptr = thrust::raw_pointer_cast(d_input.data());
    float* d_output_ptr = thrust::raw_pointer_cast(d_output.data());

    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(width * height),
        [=] __device__(int idx) {
            int x = idx % width;
            int y = idx / width;
            d_output_ptr[y * width + x] = d_input_ptr[x * height + y];
        }
    );

    thrust::copy(d_output.begin(), d_output.end(), output.begin());
}

int main() {
    const int N = 1 << 20;
    const int width = 1024;
    const int height = 1024;

    std::vector<float> a(N, 1.0f), b(N, 2.0f);
    std::vector<float> matrix(width * height, 1.0f);
    std::vector<float> transposed_matrix(width * height, 0.0f);

    // CUDA dot product
    float* d_a, * d_b, * d_result;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_a, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    dot_product_kernel<<<num_blocks, block_size>>>(d_a, d_b, d_result, N);

    float result_cuda;
    cudaMemcpy(&result_cuda, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Dot product (CUDA): " << result_cuda << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Thrust dot product
    float result_thrust = dot_product_thrust(a, b, N);
    std::cout << "Dot product (Thrust): " << result_thrust << std::endl;

    // CUDA transpose
    float* d_input, * d_output;
    cudaMalloc(&d_input, matrix.size() * sizeof(float));
    cudaMalloc(&d_output, matrix.size() * sizeof(float));
    cudaMemcpy(d_input, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(transposed_matrix.data(), d_output, matrix.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Matrix transposed (CUDA): first element: " << transposed_matrix[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    // Thrust transpose
    transpose_thrust(matrix, transposed_matrix, width, height);
    std::cout << "Matrix transposed (Thrust): first element: " << transposed_matrix[0] << std::endl;

    return 0;
}