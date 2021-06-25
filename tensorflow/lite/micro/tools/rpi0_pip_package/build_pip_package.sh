#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=$(realpath $SCRIPT_DIR/../../../../..)
TF_DIR=$ROOT_DIR/tensorflow
TFLITE_DIR=$TF_DIR/lite 
TFLM_DIR=$TFLITE_DIR/micro

export PACKAGE_VERSION=1.0.0
PYTHON=python3.7
BUILD_DIR=/tmp/tflite_micro_runtime_build
GIT_HASH=`git rev-parse HEAD`

$SCRIPT_DIR/install_tflite_micro.sh


# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/tflite_micro_runtime"
cp -r "${SCRIPT_DIR}/setup_with_binary.py" \
      "${BUILD_DIR}"
cp "${TFLM_DIR}/python/interpreter.py" "${BUILD_DIR}/tflite_micro_runtime"
cp "${TFLM_DIR}/python/image_transform.py" "${BUILD_DIR}/tflite_micro_runtime"
cp "${TFLM_DIR}/python/__init__.py" "${BUILD_DIR}/tflite_micro_runtime"

echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/tflite_micro_runtime/__init__.py"
echo "__git_version__ = '$GIT_HASH'" >> "${BUILD_DIR}/tflite_micro_runtime/__init__.py"

# Build python interpreter_wrapper.
mkdir -p "${BUILD_DIR}/cmake_build"
$SCRIPT_DIR/build_tflite_micro_wrapper.sh "${BUILD_DIR}/cmake_build" $PYTHON

LIBRARY_EXTENSION=.so


cp "${BUILD_DIR}/cmake_build/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/tflite_micro_runtime"
# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/tflite_micro_runtime/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
pushd "${BUILD_DIR}"

${PYTHON} setup_with_binary.py bdist --plat-name=linux_armv6l \
          bdist_wheel --plat-name=linux-armv6l
WHEEL_PATH=`find ${BUILD_DIR}/dist/ -name *.whl`
cp $WHEEL_PATH $ROOT_DIR
popd


echo "Output can be found here:"
find "${BUILD_DIR}/dist"


