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
add_executable(test_fgo test.cpp src/RotFGO.cpp src/TransFGO.cpp src/helper.cpp)
add_executable(reloc_one reloc_one.cpp src/RotFGO.cpp src/TransFGO.cpp src/helper.cpp)
target_link_libraries(test_fgo Eigen3::Eigen)
target_link_libraries(reloc_one Eigen3::Eigen)

# Enable comprehensive SIMD optimizations with architecture-specific flags
if(MSVC)
    # set common SIMD and warning flags for both test_fgo and reloc_one
    set(MSVC_SIMD_FLAGS
        /arch:AVX512         # Enable AVX-512 (if supported)
        /arch:AVX2           # Enable AVX2 
        /favor:INTEL64       # Optimize for Intel 64-bit
        /fp:fast             # Fast floating point model
        /W4                  # Warning level 4
    )
    target_compile_options(test_fgo PRIVATE ${MSVC_SIMD_FLAGS})
    target_compile_options(reloc_one PRIVATE ${MSVC_SIMD_FLAGS})
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(GCC_SIMD_FLAGS
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
    target_compile_options(test_fgo PRIVATE ${GCC_SIMD_FLAGS})
    target_compile_options(reloc_one PRIVATE ${GCC_SIMD_FLAGS})
endif()

# Define M_PI if not defined
target_compile_definitions(test_fgo PRIVATE _USE_MATH_DEFINES)
target_compile_definitions(reloc_one PRIVATE _USE_MATH_DEFINES)
