#include <iostream>
#include <cmath>

__global__
void productoExternoKernel(float *A, const float *B, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < m; ++k)
        {
            sum += A[row * m + k] * B[k * n + col];
        }
        A[row * n + col] = sum; // Store the result back in A
    }
}

extern "C" void productoExterno(const float *A, const float *B, int m, int n)
{
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    productoExternoKernel<<<grid, block>>>(A, B, m, n);
}