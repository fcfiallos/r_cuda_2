cmake_minimum_required(VERSION 3.16)
 
project(Cuda01)
 
enable_language(CXX CUDA)
 
find_package(fmt CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Stb REQUIRED)

include_directories(${STB_INCLUDE_DIR})

add_executable(cuda_fractal
    main.cpp
    kernel.cu
)
 
target_include_directories(cuda_fractal PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
 
target_link_libraries(cuda_fractal PRIVATE
    fmt::fmt
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::cudart
)