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
#include "tensorflow/lite/micro/python/interpreter_wrapper/interpreter_wrapper.h"

#include <stdarg.h>
#include <cstdio>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <complex>

#include "tensorflow/lite/micro/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/micro/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/micro/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/micro/python/interpreter_wrapper/model_builder.h"


#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_TENSOR_BOUNDS_CHECK(i)                                    \
  if (i >= this->NumTensors() || i < 0) {                         \
    PyErr_Format(PyExc_ValueError,                                          \
                 "Invalid tensor index %d exceeds max tensor index %lu", i, \
                 this->NumTensors());                             \
    return nullptr;                                                         \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace interpreter_wrapper {

namespace {

using python_utils::PyDecrefDeleter;


PyObject* PyArrayFromFloatVector(const float* data, npy_intp size) {
  void* pydata = malloc(size * sizeof(float));
  memcpy(pydata, data, size * sizeof(float));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_FLOAT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

PyObject* PyArrayFromIntVector(const int* data, npy_intp size) {
  void* pydata = malloc(size * sizeof(int));
  memcpy(pydata, data, size * sizeof(int));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_INT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

PyObject* PyTupleFromQuantizationParam(const TfLiteQuantizationParams& param) {
  PyObject* result = PyTuple_New(2);
  PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(param.scale));
  PyTuple_SET_ITEM(result, 1, PyLong_FromLong(param.zero_point));
  return result;
}

TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes) {
  // TODO(levp): remove the default case so that new types produce compilation
  // error.
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float);
      break;
    case kTfLiteInt32:
      *bytes = sizeof(int32_t);
      break;
    case kTfLiteUInt32:
      *bytes = sizeof(uint32_t);
      break;
    case kTfLiteUInt8:
      *bytes = sizeof(uint8_t);
      break;
    case kTfLiteInt64:
      *bytes = sizeof(int64_t);
      break;
    case kTfLiteUInt64:
      *bytes = sizeof(uint64_t);
      break;
    case kTfLiteBool:
      *bytes = sizeof(bool);
      break;
    case kTfLiteComplex64:
      *bytes = sizeof(std::complex<float>);
      break;
    case kTfLiteComplex128:
      *bytes = sizeof(std::complex<double>);
      break;
    case kTfLiteInt16:
      *bytes = sizeof(int16_t);
      break;
    case kTfLiteInt8:
      *bytes = sizeof(int8_t);
      break;
    case kTfLiteFloat16:
      *bytes = sizeof(TfLiteFloat16);
      break;
    case kTfLiteFloat64:
      *bytes = sizeof(double);
      break;
    default:
      if (context) {
        context->ReportError(
            context,
            "Type %d is unsupported. Only float16, float32, float64, int8, "
            "int16, int32, int64, uint8, uint64, bool, complex64 and "
            "complex128 supported currently.",
            type);
      }
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace


MicroInterpreterWrapper::MicroInterpreterWrapper(std::unique_ptr<Model> model,
                     std::unique_ptr<PythonErrorReporter> error_reporter,
                     MicroInterpreter *interpreter, 
                     AllOpsResolver* resolver,
                     uint8_t* tensor_arena)
    : model_(std::move(model)),
      error_reporter_(std::move(error_reporter)),
      resolver_(resolver),
      interpreter_(interpreter),
      tensor_arena_(tensor_arena) {}

MicroInterpreterWrapper::~MicroInterpreterWrapper() {
  delete interpreter_;
}

PyObject* MicroInterpreterWrapper::AllocateTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  Py_RETURN_NONE;
}

PyObject* MicroInterpreterWrapper::Invoke() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  // Release the GIL so that we can run multiple interpreters in parallel
  TfLiteStatus status_code = kTfLiteOk;
  Py_BEGIN_ALLOW_THREADS;  // To return can happen between this and end!
  status_code = interpreter_->Invoke();
  Py_END_ALLOW_THREADS;

  TFLITE_PY_CHECK(
      status_code);  // don't move this into the Py_BEGIN/Py_End block

  Py_RETURN_NONE;
}

PyObject* MicroInterpreterWrapper::InputIndices() const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  PyObject* np_array = PyArrayFromIntVector(interpreter_->inputs().data(),
                                            interpreter_->inputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* MicroInterpreterWrapper::OutputIndices() const {
  PyObject* np_array = PyArrayFromIntVector(interpreter_->outputs().data(),
                                            interpreter_->outputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

int MicroInterpreterWrapper::NumTensors() const {
  if (!interpreter_) {
    return 0;
  }
  return interpreter_->tensors_size();
}

std::string MicroInterpreterWrapper::TensorName(int i) const {
  if (!interpreter_ || i >= this->NumTensors() || i < 0) {
    return "";
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return tensor->name ? tensor->name : "";
}

PyObject* MicroInterpreterWrapper::TensorType(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->type == kTfLiteNoType) {
    PyErr_Format(PyExc_ValueError, "Tensor with no type found.");
    return nullptr;
  }

  int code = python_utils::TfLiteTypeToPyArrayType(tensor->type);
  if (code == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid tflite type code %d", code);
    return nullptr;
  }
  return PyArray_TypeObjectFromType(code);
}

PyObject* MicroInterpreterWrapper::TensorSize(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->dims == nullptr) {
    PyErr_Format(PyExc_ValueError, "Tensor with no shape found.");
    return nullptr;
  }
  PyObject* np_array =
      PyArrayFromIntVector(tensor->dims->data, tensor->dims->size);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* MicroInterpreterWrapper::TensorQuantization(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);
  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return PyTupleFromQuantizationParam(tensor->params);
}

PyObject* MicroInterpreterWrapper::TensorQuantizationParameters(int i) const {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);
  const TfLiteTensor* tensor = interpreter_->tensor(i);
  const TfLiteQuantization quantization = tensor->quantization;
  float* scales_data = nullptr;
  int32_t* zero_points_data = nullptr;
  int32_t scales_size = 0;
  int32_t zero_points_size = 0;
  int32_t quantized_dimension = 0;
  if (quantization.type == kTfLiteAffineQuantization) {
    const TfLiteAffineQuantization* q_params =
        reinterpret_cast<const TfLiteAffineQuantization*>(quantization.params);
    if (q_params->scale) {
      scales_data = q_params->scale->data;
      scales_size = q_params->scale->size;
    }
    if (q_params->zero_point) {
      zero_points_data = q_params->zero_point->data;
      zero_points_size = q_params->zero_point->size;
    }
    quantized_dimension = q_params->quantized_dimension;
  }
  PyObject* scales_array = PyArrayFromFloatVector(scales_data, scales_size);
  PyObject* zero_points_array =
      PyArrayFromIntVector(zero_points_data, zero_points_size);

  PyObject* result = PyTuple_New(3);
  PyTuple_SET_ITEM(result, 0, scales_array);
  PyTuple_SET_ITEM(result, 1, zero_points_array);
  PyTuple_SET_ITEM(result, 2, PyLong_FromLong(quantized_dimension));
  return result;
}

PyObject* MicroInterpreterWrapper::SetTensor(int i, PyObject* value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  TfLiteTensor* tensor = interpreter_->tensor(i);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Got value of type %s"
                 " but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), i, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension mismatch."
                 " Got %d"
                 " but expected %d for input %d.",
                 PyArray_NDIM(array), tensor->dims->size, i);
    return nullptr;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Dimension mismatch."
                   " Got %ld"
                   " but expected %d for dimension %d of input %d.",
                   PyArray_SHAPE(array)[j], tensor->dims->data[j], j, i);
      return nullptr;
    }
  }

  if (tensor->type != kTfLiteString) {
    if (tensor->data.raw == nullptr) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor:"
                   " Tensor is unallocated. Try calling allocate_tensors()"
                   " first");
      return nullptr;
    }

    size_t size = PyArray_NBYTES(array);
    if (size != tensor->bytes) {
      PyErr_Format(PyExc_ValueError,
                   "numpy array had %zu bytes but expected %zu bytes.", size,
                   tensor->bytes);
      return nullptr;
    }
    memcpy(tensor->data.raw, PyArray_DATA(array), size);
  } else {
      PyErr_Format(PyExc_ValueError,
                   "Cannot write string dtype tensor");
      return nullptr;
  }
  Py_RETURN_NONE;
}

namespace {

// Checks to see if a tensor access can succeed (returns nullptr on error).
// Otherwise returns Py_None.
PyObject* CheckGetTensorArgs(MicroInterpreter* interpreter_, int tensor_index,
                             TfLiteTensor** tensor, int* type_num) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  //TFLITE_PY_TENSOR_BOUNDS_CHECK(tensor_index);

  *tensor = interpreter_->tensor(tensor_index);
  if ((*tensor)->bytes == 0) {
    PyErr_SetString(PyExc_ValueError, "Invalid tensor size.");
    return nullptr;
  }

  *type_num = python_utils::TfLiteTypeToPyArrayType((*tensor)->type);
  if (*type_num == -1) {
    PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
    return nullptr;
  }

  if (!(*tensor)->data.raw) {
    PyErr_SetString(PyExc_ValueError,
                    "Tensor data is null."
                    " Run allocate_tensors() first");
    return nullptr;
  }

  Py_RETURN_NONE;
}

}  // namespace

PyObject* MicroInterpreterWrapper::GetTensor(int i) const {
  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;

  PyObject* check_result =
      CheckGetTensorArgs(interpreter_, i, &tensor, &type_num);
  if (check_result == nullptr) return check_result;
  Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  if (tensor->type != kTfLiteString && tensor->type != kTfLiteResource &&
      tensor->type != kTfLiteVariant) {
    // Make a buffer copy but we must tell Numpy It owns that data or else
    // it will leak.
    void* data = malloc(tensor->bytes);
    if (!data) {
      PyErr_SetString(PyExc_ValueError, "Malloc to copy tensor failed.");
      return nullptr;
    }
    memcpy(data, tensor->data.raw, tensor->bytes);
    PyObject* np_array;
    if (tensor->sparsity == nullptr) {
      np_array =
          PyArray_SimpleNewFromData(dims.size(), dims.data(), type_num, data);
    } else {
      std::vector<npy_intp> sparse_buffer_dims(1);
      size_t size_of_type;
      if (GetSizeOfType(nullptr, tensor->type, &size_of_type) != kTfLiteOk) {
        PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
        free(data);
        return nullptr;
      }
      sparse_buffer_dims[0] = tensor->bytes / size_of_type;
      np_array = PyArray_SimpleNewFromData(
          sparse_buffer_dims.size(), sparse_buffer_dims.data(), type_num, data);
    }
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(np_array),
                        NPY_ARRAY_OWNDATA);
    return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
  } else {
        PyErr_Format(PyExc_ValueError,
                     "Tensor dtype not supported");
        return nullptr;
  }
}

PyObject* MicroInterpreterWrapper::tensor(PyObject* base_object, int i) {
  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;

  PyObject* check_result =
      CheckGetTensorArgs(interpreter_, i, &tensor, &type_num);
  if (check_result == nullptr) return check_result;
  Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  PyArrayObject* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(
          dims.size(), dims.data(), type_num, tensor->data.raw));
  Py_INCREF(base_object);  // SetBaseObject steals, so we need to add.
  PyArray_SetBaseObject(np_array, base_object);
  return PyArray_Return(np_array);
}

MicroInterpreterWrapper* MicroInterpreterWrapper::CreateWrapperCPPFromFile(
    const char* model_path, size_t tensor_arena_size, std::string* error_msg) {
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  std::unique_ptr<MicroInterpreterWrapper::Model> model =
      Model::BuildFromFile(model_path, error_reporter.get());
  if(tensor_arena_size == 0 || tensor_arena_size > 64*1024*1024) {
    *error_msg = "Invalid tensor_arena_size";
    return nullptr;
  }
  if(!model)
  {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  python::ImportNumpy();

  auto resolver = new AllOpsResolver();
  auto tensor_arena = new uint8_t[tensor_arena_size];

  auto interpreter = new MicroInterpreter( 
    model->GetModel(), 
    *resolver, 
    tensor_arena, tensor_arena_size, 
    error_reporter.get()
  );

  auto wrapper = new MicroInterpreterWrapper(
    std::move(model), std::move(error_reporter),
    interpreter, resolver, tensor_arena);
 
  return wrapper;
}


PyObject* MicroInterpreterWrapper::ResetVariableTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
