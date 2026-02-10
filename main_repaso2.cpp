#include <iostream>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <vector>

#define N = 1024; // Tamaño de las matrices

extern "C" void productoExterno(const float *A, const float *B,  int m, int n);

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    cudaSetDevice(0);
    std::vector<float> A(N * N, 1.0f); // Matriz A de tamaño N x N
    std::vector<float> B(N * N, 1.0f); // Matriz B de tamaño N x N

    std::vector<float> C(N * N, 0.0f); // Matriz C para almacenar el resultado
 float *d_A, *d_B; // Punteros para las matrices en la GPU
    cudaMalloc(&d_A, N * N * sizeof(float)); // Reservar memoria para
    cudaMalloc(&d_B, N * N * sizeof(float)); // Reservar memoria para B
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice); // Copiar A a la GPU
    cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice); // Copiar B a la GPU
    productoExterno(d_A, d_B, N, N); // Llamar a la función de producto externo en la GPU
    cudaMemcpy(C.data(), d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost); // Copiar el resultado de vuelta a la CPU
    cudaFree(d_A); // Liberar memoria de A en la GPU
    cudaFree(d_B); // Liberar memoria de B en la GPU
    return 0;
}
