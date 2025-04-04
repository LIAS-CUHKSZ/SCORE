cmake_minimum_required(VERSION 3.0.0)
project(FGO_demo)

## Compile as C++17
add_compile_options(-std=c++17)

# Add OpenMP support
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message(STATUS "OpenMP found")
endif()

find_package(OpenCV)
find_package(Eigen3 REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    include
)

# add macro USE_OMP
add_definitions(-DUSE_OMP)

add_library(BnB_module SHARED src/BnB.cpp)
add_library(helper SHARED src/helper.cpp)
add_executable(FGO_demo FGO_demo.cpp src/BnB.cpp)

# Properly link OpenMP
target_link_libraries(helper OpenMP::OpenMP_CXX)
target_link_libraries(BnB_module helper OpenMP::OpenMP_CXX)
target_link_libraries(FGO_demo 
    BnB_module 
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)


# include_directories(/usr/include/vtk-9.1)

# link_directories(${PCL_LIBRARY_DIRS})

# add_definitions(${PCL_DEFINITIONS})

# add_executable(main main.cpp linefunc.cpp BnB_rot.cpp project.cpp read_txt.cpp extract_line.cpp common.cpp)
# add_executable(convert convert.cpp)
# target_link_libraries(main 
#   ${OpenCV_LIBS} ${PCL_LIBRARIES} pcl_visualization
# )
# target_link_libraries(convert 
#   ${OpenCV_LIBS} ${PCL_LIBRARIES} pcl_visualization
# )
