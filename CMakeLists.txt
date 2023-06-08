cmake_minimum_required(VERSION 3.9.1)
project(yolov5_ort)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(ONNXRUNTIME_INCLUDE_DIRS "/root/yolov5-engine/onnxruntime-linux-x64-gpu-1.7.0/include")
set(ONNXRUNTIME_LIBS "/root/yolov5-engine/onnxruntime-linux-x64-gpu-1.7.0/lib/libonnxruntime.so")
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} "src/main.cpp")

target_link_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBS} ${OpenCV_LIBS})