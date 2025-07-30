cmake_minimum_required(VERSION 3.16)
project(FGO_CPP)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_BUILD_TYPE "Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

# include directories
include_directories(
    include
)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(test_rotfgo src/test.cpp src/RotFGO.cpp src/TransFGO.cpp src/helper.cpp)
add_executable(experiment src/Experiment.cpp src/RotFGO.cpp src/TransFGO.cpp src/helper.cpp)
target_link_libraries(test_rotfgo Eigen3::Eigen)
target_link_libraries(experiment Eigen3::Eigen)

# Enable comprehensive SIMD optimizations with architecture-specific flags
if(MSVC)
    # MSVC SIMD flags
    target_compile_options(test_rotfgo PRIVATE 
        /arch:AVX512         # Enable AVX-512 (if supported)
        /arch:AVX2           # Enable AVX2 
        /favor:INTEL64       # Optimize for Intel 64-bit
        /fp:fast             # Fast floating point model
        /W4                  # Warning level 4
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # GCC/Clang comprehensive SIMD flags
    target_compile_options(test_rotfgo PRIVATE 
        -Wall -Wextra -pedantic
        # SIMD instruction sets (progressive capability)
        -msse4.1             # SSE4.1 instructions
        -msse4.2             # SSE4.2 instructions
        -mavx                # AVX instructions
        -mavx2               # AVX2 instructions
        -mfma                # Fused multiply-add
        -mbmi                # Bit manipulation instructions
        -mbmi2               # Bit manipulation instructions 2
        -mlzcnt              # Leading zero count
        -mpopcnt             # Population count
        # Performance optimizations
        -mtune=native        # Tune for the current CPU
        -march=native        # Generate code for current CPU architecture
        -funroll-loops       # Unroll loops for better performance
        -ftree-vectorize     # Enable auto-vectorization        
    )
endif()

# Define M_PI if not defined
target_compile_definitions(test_rotfgo PRIVATE _USE_MATH_DEFINES)
