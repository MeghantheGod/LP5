#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

int main() {
    int N = 512;
    int size = N * N * sizeof(int);
    int *A = nullptr, *B = nullptr, *C = nullptr;
    int *dev_A = nullptr, *dev_B = nullptr, *dev_C = nullptr;

    // Allocate pinned memory on host
    if (cudaMallocHost((void**)&A, size) != cudaSuccess ||
        cudaMallocHost((void**)&B, size) != cudaSuccess ||
        cudaMallocHost((void**)&C, size) != cudaSuccess) 
    {
        std::cerr << "Failed to allocate pinned host memory!" << std::endl;
        return -1;
    }

    // Allocate memory on device
    if (cudaMalloc((void**)&dev_A, size) != cudaSuccess ||
        cudaMalloc((void**)&dev_B, size) != cudaSuccess ||
        cudaMalloc((void**)&dev_C, size) != cudaSuccess) 
    {
        std::cerr << "Failed to allocate device memory!" << std::endl;
        return -1;
    }

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i * N + j;
            B[i * N + j] = j * N + i;
        }
    }

    // Copy matrices to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    // Synchronize to wait for kernel completion
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print a portion of the result matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
