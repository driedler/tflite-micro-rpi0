/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
#define TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Place `<locale>` before <Python.h> to avoid build failures in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/python/interpreter_wrapper/allocation.h"


// We forward declare TFLite classes here to avoid exposing them to SWIG.
namespace tflite {

class FlatBufferModel;

namespace interpreter_wrapper {

class PythonErrorReporter;

class InterpreterWrapper {
 public:
  using Model = FlatBufferModel;

  // SWIG caller takes ownership of pointer.
  static InterpreterWrapper* CreateWrapperCPPFromFile(
      const char* model_path, size_t tensor_arena_size, std::string* error_msg);

  ~InterpreterWrapper();
  PyObject* AllocateTensors();
  PyObject* Invoke();

  PyObject* InputIndices() const;
  PyObject* OutputIndices() const;
  PyObject* ResizeInputTensor(int i, PyObject* value, bool strict);

  int NumTensors() const;
  std::string TensorName(int i) const;
  PyObject* TensorType(int i) const;
  PyObject* TensorSize(int i) const;
  //PyObject* TensorSparsityParameters(int i) const;
  // Deprecated in favor of TensorQuantizationScales, below.
  PyObject* TensorQuantization(int i) const;
  PyObject* TensorQuantizationParameters(int i) const;
  PyObject* SetTensor(int i, PyObject* value);
  PyObject* GetTensor(int i) const;
  // Returns a reference to tensor index i as a numpy array. The base_object
  // should be the interpreter object providing the memory.
  PyObject* tensor(PyObject* base_object, int i);

  PyObject* ResetVariableTensors();

  // Experimental and subject to change.
  //
  // Returns a pointer to the underlying interpreter.
  MicroInterpreter* interpreter() { return interpreter_; }

 private:

  InterpreterWrapper(std::unique_ptr<Model> model,
                     std::unique_ptr<PythonErrorReporter> error_reporter,
                     MicroInterpreter *interpreter, 
                     AllOpsResolver* resolver,
                     uint8_t* tensor_arena);

  // InterpreterWrapper is not copyable or assignable. 
  InterpreterWrapper() = delete;
  InterpreterWrapper(const InterpreterWrapper& rhs) = delete;


  // The public functions which creates `InterpreterWrapper` should ensure all
  // these member variables are initialized successfully. Otherwise it should
  // report the error and return `nullptr`.
  MicroInterpreter* interpreter_;
  const std::unique_ptr<Model> model_;
  const std::unique_ptr<PythonErrorReporter> error_reporter_;
  std::unique_ptr<AllOpsResolver> resolver_;
  std::unique_ptr<uint8_t> tensor_arena_;
};

}  // namespace interpreter_wrapper
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
