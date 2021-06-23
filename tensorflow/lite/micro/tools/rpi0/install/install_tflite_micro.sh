#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR=/workspace/tflite-micro-rpi0/tensorflow
TFLITE_DIR=$TF_DIR/lite 
TFLM_DIR=$TF_DIR/lite/micro


mkdir -p /workspace

cd workspace
git clone https://github.com/driedler/tflite-micro-rpi0.git


$TFLITE_DIR/tools/cmake/download_toolchains.sh rpi0 
$TFLM_DIR/tools/make/gemmlowp_download.sh  $TFLM_DIR/tools/make/downloads
$TFLM_DIR/tools/make/flatbuffers_download.sh $TFLM_DIR/tools/make/downloads
$TFLM_DIR/tools/make/ruy_download.sh $TFLM_DIR/tools/make/downloads




