#include <iostream>
#include <cuda_runtime.h>
#include <fmt/core.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern "C" void applyBlur(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels);

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    cudaSetDevice(0);

    int width, height, channels;
    unsigned char *inputImage = stbi_load("input.jpg", &width, &height, &channels, 0);
    if (inputImage == nullptr) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;
    unsigned char *outputImage = new unsigned char[width * height * channels];
    cudaMalloc(&d_inputImage, width * height * channels);
    cudaMalloc(&d_outputImage, width * height * channels);
    cudaMemcpy(d_inputImage, inputImage, width * height * channels, cudaMemcpyHostToDevice);

    applyBlur(d_inputImage, d_outputImage, width, height, channels);

    cudaMemcpy(outputImage, d_outputImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("output.jpg", width, height, channels, outputImage, 100);
    stbi_image_free(inputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    delete[] outputImage;
    return 0;
}
