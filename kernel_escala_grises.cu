#include <iostream>
#include <cmath>

__global__
void escalaGrisesKernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    unsigned char r = inputImage[idx];
    unsigned char g = inputImage[idx + 1];
    unsigned char b = inputImage[idx + 2];

    // Convert to grayscale using luminosity method
    unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

    outputImage[idx] = gray;
    outputImage[idx + 1] = gray;
    outputImage[idx + 2] = gray;
    if (channels == 4)
        outputImage[idx + 3] = inputImage[idx + 3]; // Preserve alpha channel
}

extern "C" void applyGrayscale(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels)
{
    dim3 block(16, 16); // 256 threads per block (16x16)
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    escalaGrisesKernel<<<grid, block>>>(inputImage, outputImage, width, height, channels);
}