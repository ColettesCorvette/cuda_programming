cmake_minimum_required(VERSION 3.18)
project(cuda_programming LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(idxing src/01_idxing.cu)

add_executable(raytracer src/raytracer/main.cc)

target_compile_options(raytracer PRIVATE -Wall -Wextra -Wpedantic)

#Sanitizers ASan/UBSan pour prévenir les bugs mémoire avant gdb 
#Tu lances ton programme normalement ; 
# au premier use-after-free, débordement de tableau ou overflow signé, il crash avec une stack trace annotée indiquant la ligne exacte. 
#En pratique : un build Debug avec sanitizers pour développer, un build RelWithDebInfo sans pour profiler.
target_compile_options(raytracer PRIVATE -fsanitize=address,undefined -g)
target_link_options(raytracer PRIVATE -fsanitize=address,undefined)