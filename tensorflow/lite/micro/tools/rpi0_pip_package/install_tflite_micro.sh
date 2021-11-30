#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR=$(realpath $SCRIPT_DIR/../../../..)
TFLITE_DIR=$TF_DIR/lite 
TFLM_DIR=$TF_DIR/lite/micro

echo "Install RPI0 toolchain ..."
$TFLITE_DIR/tools/cmake/download_toolchains.sh rpi0 

mkdir -p $TFLM_DIR/tools/make/downloads
$TFLM_DIR/tools/make/gemmlowp_download.sh  $TFLM_DIR/tools/make/downloads
$TFLM_DIR/tools/make/flatbuffers_download.sh $TFLM_DIR/tools/make/downloads
$TFLM_DIR/tools/make/ruy_download.sh $TFLM_DIR/tools/make/downloads
$TFLM_DIR/tools/make/ext_libs/cmsis_download.sh $TFLM_DIR/tools/make/downloads

python3 $SCRIPT_DIR/patch_cmsis.py


