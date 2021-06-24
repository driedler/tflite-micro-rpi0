cmake_minimum_required(VERSION 3.16)

project(tflite_runtime_rpi0 
VERSION 1.0.0)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(tensorflow/lite/micro)