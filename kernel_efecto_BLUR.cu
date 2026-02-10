#include <iostream>
#include <cmath>

__global__ 
void blurKernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    // Simple blur kernel (3x3)
    float r = 0.0f, g = 0.0f, b = 0.0f;
    int count = 0;

    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                int nidx = (ny * width + nx) * channels;
                r += inputImage[nidx];
                g += inputImage[nidx + 1];
                b += inputImage[nidx + 2];
                count++;
            }
        }
    }

    outputImage[idx] = static_cast<unsigned char>(r / count);
    outputImage[idx + 1] = static_cast<unsigned char>(g / count);
    outputImage[idx + 2] = static_cast<unsigned char>(b / count);
    if (channels == 4)
        outputImage[idx + 3] = inputImage[idx + 3]; // Preserve alpha channel
}

extern "C" void applyBlur(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels)
{
    dim3 block(16, 16); // 256 threads per block (16x16)
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    blurKernel<<<grid, block>>>(inputImage, outputImage, width, height, channels);

}
