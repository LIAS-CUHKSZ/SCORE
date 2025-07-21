cmake_minimum_required(VERSION 3.12)
project(RotFGO_CPP)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_BUILD_TYPE "Debug")
# if not debug, use release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test_rotfgo 
    test.cpp 
    RotFGO.cpp
)

# Link libraries
target_link_libraries(test_rotfgo Eigen3::Eigen)

# Set compile flags
if(MSVC)
    target_compile_options(test_rotfgo PRIVATE /W4)
else()
    target_compile_options(test_rotfgo PRIVATE -Wall -Wextra -pedantic)
endif()

# Define M_PI if not defined
target_compile_definitions(test_rotfgo PRIVATE _USE_MATH_DEFINES)