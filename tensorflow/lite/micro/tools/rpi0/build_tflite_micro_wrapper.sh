#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../../..)
TF_DIR=$ROOT_DIR/tensorflow
TFLITE_DIR=$TF_DIR/lite 
TFLM_DIR=$TF_DIR/lite/micro

cmake -DCMAKE_TOOLCHAIN_FILE=$TFLITE_DIR/tools/cmake/rpi0_toolchain.cmake -S $ROOT_DIR -B $ROOT_DIR/build 
cmake --build $ROOT_DIR/build --target _pywrap_tensorflow_interpreter_wrapper -- -j16
