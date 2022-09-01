cmake_minimum_required(VERSION 3.18)

project(CUDA_raytracer LANGUAGES CUDA CXX)

find_package( OpenCV REQUIRED )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(trace main.cu
                     export.cpp export.h
                     raytracing.cu raytracing.h
                     geometry.cu geometry.h
                     utils.h)
set_target_properties(trace PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            POSITION_INDEPENDENT_CODE ON
                            CUDA_ARCHITECTURES "61")
target_compile_features(trace PUBLIC cuda_std_17)
target_include_directories(trace PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trace PUBLIC ${OpenCV_LIBS})




