#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../../..)
TF_DIR=$ROOT_DIR/tensorflow
TFLITE_DIR=$TF_DIR/lite 
TFLM_DIR=$TFLITE_DIR/micro


BUILD_DIR=${1:-$ROOT_DIR/build}
PYTHON=${2:-python3.7}

cmake -DCMAKE_TOOLCHAIN_FILE=$TFLITE_DIR/tools/cmake/rpi0_toolchain.cmake -DPYTHON=${PYTHON} -S $ROOT_DIR -B $BUILD_DIR 
cmake --build $BUILD_DIR --config Release --target _pywrap_tflm_interpreter_wrapper -j8
