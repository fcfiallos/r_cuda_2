#include <iostream>
#include <cmath>

__global__
void productoMatricesKernel(const float *A, const float *B, float *C, int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

extern "C" void productoMatrices(const float *A, const float *B, float *C, int m, int n, int p)
{
    dim3 block(16, 16);
    dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    productoMatricesKernel<<<grid, block>>>(A, B, C, m, n, p);
}