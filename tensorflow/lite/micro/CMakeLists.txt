#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the Tensorflow Lite runtime.
#
# WARNING: This is an experimental that is subject to change.
# This has only been tested on Windows, Linux and macOS.
#
# The following are not currently supported:
# - iOS
# - Micro backend
# - Tests
# - Many features in experimental
# - Host Tools (i.e conversion / analysis tools etc.)

cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Double colon in target name means ALIAS or IMPORTED target.
cmake_policy(SET CMP0028 NEW)
# Enable MACOSX_RPATH (@rpath) for built dynamic libraries.
cmake_policy(SET CMP0042 NEW)
project(tensorflow-lite-micro C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../.."
    ABSOLUTE
  )
endif()
set(TF_SOURCE_DIR "${TENSORFLOW_SOURCE_DIR}/tensorflow")
set(TFLITE_SOURCE_DIR "${TF_SOURCE_DIR}/lite")
set(TFLM_SOURCE_DIR "${TFLITE_SOURCE_DIR}/micro")
set(CMAKE_MODULE_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_MODULE_PATH}
)
set(CMAKE_PREFIX_PATH
  "${TFLITE_SOURCE_DIR}/tools/cmake/modules"
  ${CMAKE_PREFIX_PATH}
)
include(CMakeDependentOption)
set(CMAKE_CXX_STANDARD 14)  # Some components require C++14.
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_INSTALL OFF)
set(FLATBUFFERS_STATIC_FLATC ON)
set(FLATBUFFERS_BUILD_FLATHASH OFF)
set(FLATBUFFERS_LIBCXX_WITH_CLANG OFF)
add_subdirectory(tools/make/downloads/flatbuffers)

execute_process(COMMAND python3.7 -c "from sysconfig import get_paths as gp; print(gp()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3.7 -c "import pybind11; print (pybind11.get_include())" OUTPUT_VARIABLE PYBIND11_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3.7 -c "import numpy; print (numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "PYTHON_INCLUDE=${PYTHON_INCLUDE}")
message(STATUS "PYBIND11_INCLUDE=${PYBIND11_INCLUDE}")
message(STATUS "NUMPY_INCLUDE=${NUMPY_INCLUDE}")


# TFLite library
add_library(tensorflow-lite-micro)

target_sources(tensorflow-lite-micro
PRIVATE
  ${TFLITE_SOURCE_DIR}/c/common.c 
  ${TFLITE_SOURCE_DIR}/core/api/flatbuffer_conversions.cc 
  ${TFLITE_SOURCE_DIR}/core/api/op_resolver.cc 
  ${TFLITE_SOURCE_DIR}/core/api/tensor_utils.cc
  ${TFLITE_SOURCE_DIR}/core/api/error_reporter.cc
  ${TFLITE_SOURCE_DIR}/schema/schema_utils.cc
  ${TFLITE_SOURCE_DIR}/kernels/internal/quantization_util.cc
  ${TFLITE_SOURCE_DIR}/kernels/kernel_util.cc
  ${TFLM_SOURCE_DIR}/all_ops_resolver.cc 
  ${TFLM_SOURCE_DIR}/debug_log.cc 
  ${TFLM_SOURCE_DIR}/memory_helpers.cc 
  ${TFLM_SOURCE_DIR}/micro_allocator.cc 
  ${TFLM_SOURCE_DIR}/micro_error_reporter.cc 
  ${TFLM_SOURCE_DIR}/micro_graph.cc 
  ${TFLM_SOURCE_DIR}/micro_interpreter.cc 
  ${TFLM_SOURCE_DIR}/micro_profiler.cc 
  ${TFLM_SOURCE_DIR}/micro_string.cc 
  ${TFLM_SOURCE_DIR}/micro_time.cc 
  ${TFLM_SOURCE_DIR}/micro_utils.cc 
  ${TFLM_SOURCE_DIR}/memory_planner/greedy_memory_planner.cc 
  ${TFLM_SOURCE_DIR}/memory_planner/linear_memory_planner.cc
  ${TFLM_SOURCE_DIR}/simple_memory_allocator.cc 
  ${TFLM_SOURCE_DIR}/kernels/activations.cc 
  ${TFLM_SOURCE_DIR}/kernels/add_n.cc 
  ${TFLM_SOURCE_DIR}/kernels/add.cc 
  ${TFLM_SOURCE_DIR}/kernels/arg_min_max.cc 
  ${TFLM_SOURCE_DIR}/kernels/batch_to_space_nd.cc 
  ${TFLM_SOURCE_DIR}/kernels/cast.cc 
  ${TFLM_SOURCE_DIR}/kernels/ceil.cc
  ${TFLM_SOURCE_DIR}/kernels/circular_buffer.cc 
  ${TFLM_SOURCE_DIR}/kernels/comparisons.cc 
  ${TFLM_SOURCE_DIR}/kernels/concatenation.cc 
  ${TFLM_SOURCE_DIR}/kernels/conv_common.cc 
  ${TFLM_SOURCE_DIR}/kernels/conv.cc 
  ${TFLM_SOURCE_DIR}/kernels/cumsum.cc 
  ${TFLM_SOURCE_DIR}/kernels/depth_to_space.cc 
  ${TFLM_SOURCE_DIR}/kernels/depthwise_conv.cc 
  ${TFLM_SOURCE_DIR}/kernels/depthwise_conv_common.cc 
  ${TFLM_SOURCE_DIR}/kernels/dequantize.cc 
  ${TFLM_SOURCE_DIR}/kernels/detection_postprocess.cc 
  ${TFLM_SOURCE_DIR}/kernels/elementwise.cc 
  ${TFLM_SOURCE_DIR}/kernels/elu.cc 
  ${TFLM_SOURCE_DIR}/kernels/ethosu.cc 
  ${TFLM_SOURCE_DIR}/kernels/exp.cc 
  ${TFLM_SOURCE_DIR}/kernels/expand_dims.cc 
  ${TFLM_SOURCE_DIR}/kernels/fill.cc 
  ${TFLM_SOURCE_DIR}/kernels/floor_div.cc 
  ${TFLM_SOURCE_DIR}/kernels/floor_mod.cc 
  ${TFLM_SOURCE_DIR}/kernels/floor.cc 
  ${TFLM_SOURCE_DIR}/kernels/fully_connected.cc 
  ${TFLM_SOURCE_DIR}/kernels/fully_connected_common.cc 
  ${TFLM_SOURCE_DIR}/kernels/gather_nd.cc 
  ${TFLM_SOURCE_DIR}/kernels/gather.cc 
  ${TFLM_SOURCE_DIR}/kernels/hard_swish.cc 
  ${TFLM_SOURCE_DIR}/kernels/if.cc 
  ${TFLM_SOURCE_DIR}/kernels/kernel_util.cc 
  ${TFLM_SOURCE_DIR}/kernels/l2_pool_2d.cc 
  ${TFLM_SOURCE_DIR}/kernels/l2norm.cc 
  ${TFLM_SOURCE_DIR}/kernels/leaky_relu.cc 
  ${TFLM_SOURCE_DIR}/kernels/log_softmax.cc 
  ${TFLM_SOURCE_DIR}/kernels/logical.cc 
  ${TFLM_SOURCE_DIR}/kernels/logistic.cc 
  ${TFLM_SOURCE_DIR}/kernels/maximum_minimum.cc 
  ${TFLM_SOURCE_DIR}/kernels/mul.cc 
  ${TFLM_SOURCE_DIR}/kernels/neg.cc 
  ${TFLM_SOURCE_DIR}/kernels/pack.cc 
  ${TFLM_SOURCE_DIR}/kernels/pad.cc 
  ${TFLM_SOURCE_DIR}/kernels/pooling.cc 
  ${TFLM_SOURCE_DIR}/kernels/prelu.cc 
  ${TFLM_SOURCE_DIR}/kernels/quantize_common.cc 
  ${TFLM_SOURCE_DIR}/kernels/quantize.cc 
  ${TFLM_SOURCE_DIR}/kernels/reduce.cc 
  ${TFLM_SOURCE_DIR}/kernels/reshape.cc 
  ${TFLM_SOURCE_DIR}/kernels/resize_bilinear.cc 
  ${TFLM_SOURCE_DIR}/kernels/resize_nearest_neighbor.cc 
  ${TFLM_SOURCE_DIR}/kernels/round.cc 
  ${TFLM_SOURCE_DIR}/kernels/shape.cc 
  ${TFLM_SOURCE_DIR}/kernels/softmax_common.cc 
  ${TFLM_SOURCE_DIR}/kernels/softmax.cc 
  ${TFLM_SOURCE_DIR}/kernels/space_to_batch_nd.cc 
  ${TFLM_SOURCE_DIR}/kernels/split_v.cc 
  ${TFLM_SOURCE_DIR}/kernels/split.cc 
  ${TFLM_SOURCE_DIR}/kernels/squeeze.cc 
  ${TFLM_SOURCE_DIR}/kernels/strided_slice.cc 
  ${TFLM_SOURCE_DIR}/kernels/sub.cc 
  ${TFLM_SOURCE_DIR}/kernels/svdf_common.cc 
  ${TFLM_SOURCE_DIR}/kernels/svdf.cc 
  ${TFLM_SOURCE_DIR}/kernels/tanh.cc 
  ${TFLM_SOURCE_DIR}/kernels/transpose_conv.cc 
  ${TFLM_SOURCE_DIR}/kernels/transpose.cc 
  ${TFLM_SOURCE_DIR}/kernels/unpack.cc 
  ${TFLM_SOURCE_DIR}/kernels/zeros_like.cc
)
target_include_directories(tensorflow-lite-micro
  PUBLIC
    ${TENSORFLOW_SOURCE_DIR}
    ${TFLM_SOURCE_DIR}/tools/make/downloads/gemmlowp
    ${TFLM_SOURCE_DIR}/tools/make/downloads/ruy
)
target_link_libraries(tensorflow-lite-micro
  PUBLIC
    flatbuffers
)

target_compile_options(tensorflow-lite-micro
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)


add_library(_pywrap_tensorflow_interpreter_wrapper SHARED EXCLUDE_FROM_ALL)
target_sources(_pywrap_tensorflow_interpreter_wrapper
PUBLIC
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/interpreter_wrapper.cc
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/interpreter_wrapper_pybind11.cc
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/numpy.cc
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/allocation.cc
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/model_builder.cc
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/python_error_reporter.cc
  ${TFLM_SOURCE_DIR}/python/interpreter_wrapper/python_utils.cc
)

# # To remove "lib" prefix.
set_target_properties(_pywrap_tensorflow_interpreter_wrapper PROPERTIES PREFIX "")

target_include_directories(_pywrap_tensorflow_interpreter_wrapper
  PUBLIC
    ${TENSORFLOW_SOURCE_DIR}
    ${PYTHON_INCLUDE}
    ${PYBIND11_INCLUDE}
    ${NUMPY_INCLUDE}
)

target_link_libraries(_pywrap_tensorflow_interpreter_wrapper
PUBLIC
  tensorflow-lite-micro
)
target_compile_options(_pywrap_tensorflow_interpreter_wrapper
  PUBLIC ${TFLITE_TARGET_PUBLIC_OPTIONS}
  PRIVATE ${TFLITE_TARGET_PRIVATE_OPTIONS}
)
