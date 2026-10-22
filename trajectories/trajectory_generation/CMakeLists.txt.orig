cmake_minimum_required(VERSION 3.4...3.18)

project(trajectory_cpp)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -fPIC")

find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    include
)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} src/pywrap.cpp src/traj_gen.cpp src/trajectory_wrapper.cpp)

# add_executable(${PROJECT_NAME}_node src/example1_node.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
