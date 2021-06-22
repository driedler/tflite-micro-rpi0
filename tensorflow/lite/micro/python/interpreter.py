# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Python TF-Lite interpreter."""

import ctypes
import enum
import os
import platform
import sys

import numpy as np


  # This file is part of tflite_runtime package.
try:
  from tflite_micro_runtime import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper
except:
  import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper



class Interpreter(object):
  """Interpreter interface for TensorFlow Lite Models.

  This makes the TensorFlow Lite interpreter accessible in Python.
  It is possible to use this interpreter in a multithreaded Python environment,
  but you must be sure to call functions of a particular instance from only
  one thread at a time. So if you want to have 4 threads running different
  inferences simultaneously, create  an interpreter for each one as thread-local
  data. Similarly, if you are calling invoke() in one thread on a single
  interpreter but you want to use tensor() on another thread once it is done,
  you must use a synchronization primitive between the threads to ensure invoke
  has returned before calling tensor().
  """

  def __init__(self, model_path, tensor_arena_size=4*1024*1024):
    """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.
  
    Raises:
      ValueError: If the interpreter was unable to create.
    """

    if model_path:
      self._interpreter = _interpreter_wrapper.CreateWrapperFromFile(model_path, tensor_arena_size)
      if not self._interpreter:
        raise ValueError('Failed to open {}'.format(model_path))


  def __del__(self):
    # Must make sure the interpreter is destroyed before things that
    # are used by it like the delegates. NOTE this only works on CPython
    # probably.
    # TODO(b/136468453): Remove need for __del__ ordering needs of CPython
    # by using explicit closes(). See implementation of Interpreter __del__.
    self._interpreter = None

  def allocate_tensors(self):
    self._ensure_safe()
    return self._interpreter.AllocateTensors()

  def _safe_to_run(self):
    """Returns true if there exist no numpy array buffers.

    This means it is safe to run tflite calls that may destroy internally
    allocated memory. This works, because in the wrapper.cc we have made
    the numpy base be the self._interpreter.
    """
    # NOTE, our tensor() call in cpp will use _interpreter as a base pointer.
    # If this environment is the only _interpreter, then the ref count should be
    # 2 (1 in self and 1 in temporary of sys.getrefcount).
    return sys.getrefcount(self._interpreter) == 2

  def _ensure_safe(self):
    """Makes sure no numpy arrays pointing to internal buffers are active.

    This should be called from any function that will call a function on
    _interpreter that may reallocate memory e.g. invoke(), ...

    Raises:
      RuntimeError: If there exist numpy objects pointing to internal memory
        then we throw.
    """
    if not self._safe_to_run():
      raise RuntimeError("""There is at least 1 reference to internal data
      in the interpreter in the form of a numpy array or slice. Be sure to
      only hold the function returned from tensor() if you are using raw
      data access.""")



  def _get_tensor_details(self, tensor_index):
    """Gets tensor details.

    Args:
      tensor_index: Tensor index of tensor to query.

    Returns:
      A dictionary containing the following fields of the tensor:
        'name': The tensor name.
        'index': The tensor index in the interpreter.
        'shape': The shape of the tensor.
        'quantization': Deprecated, use 'quantization_parameters'. This field
            only works for per-tensor quantization, whereas
            'quantization_parameters' works in all cases.
        'quantization_parameters': The parameters used to quantize the tensor:
          'scales': List of scales (one if per-tensor quantization)
          'zero_points': List of zero_points (one if per-tensor quantization)
          'quantized_dimension': Specifies the dimension of per-axis
              quantization, in the case of multiple scales/zero_points.

    Raises:
      ValueError: If tensor_index is invalid.
    """
    tensor_index = int(tensor_index)
    tensor_name = self._interpreter.TensorName(tensor_index)
    tensor_size = self._interpreter.TensorSize(tensor_index)
    tensor_type = self._interpreter.TensorType(tensor_index)
    tensor_quantization = self._interpreter.TensorQuantization(tensor_index)
    tensor_quantization_params = self._interpreter.TensorQuantizationParameters(
        tensor_index)
    # tensor_sparsity_params = self._interpreter.TensorSparsityParameters(
    #     tensor_index)

    if not tensor_type:
      raise ValueError('Could not get tensor details')

    details = {
        'name': tensor_name,
        'index': tensor_index,
        'shape': tensor_size,
        'dtype': tensor_type,
        'quantization': tensor_quantization,
        'quantization_parameters': {
            'scales': tensor_quantization_params[0],
            'zero_points': tensor_quantization_params[1],
            'quantized_dimension': tensor_quantization_params[2],
        },
        #'sparsity_parameters': tensor_sparsity_params
    }

    return details


  def get_tensor_details(self):
    """Gets tensor details for every tensor with valid tensor details.

    Tensors where required information about the tensor is not found are not
    added to the list. This includes temporary tensors without a name.

    Returns:
      A list of dictionaries containing tensor information.
    """
    tensor_details = []
    for idx in range(self._interpreter.NumTensors()):
      try:
        tensor_details.append(self._get_tensor_details(idx))
      except ValueError:
        pass
    return tensor_details

  def get_input_details(self):
    """Gets model input details.

    Returns:
      A list of input details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.InputIndices()
    ]

  def set_tensor(self, tensor_index, value):
    """Sets the value of the input tensor.

    Note this copies data in `value`.

    If you want to avoid copying, you can use the `tensor()` function to get a
    numpy buffer pointing to the input buffer in the tflite interpreter.

    Args:
      tensor_index: Tensor index of tensor to set. This value can be gotten from
        the 'index' field in get_input_details.
      value: Value of tensor to set.

    Raises:
      ValueError: If the interpreter could not set the tensor.
    """
    self._interpreter.SetTensor(tensor_index, value)

 
  def get_output_details(self):
    """Gets model output details.

    Returns:
      A list of output details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.OutputIndices()
    ]


  def get_tensor(self, tensor_index):
    """Gets the value of the output tensor (get a copy).

    If you wish to avoid the copy, use `tensor()`. This function cannot be used
    to read intermediate results.

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
        the 'index' field in get_output_details.

    Returns:
      a numpy array.
    """
    return self._interpreter.GetTensor(tensor_index)

  def tensor(self, tensor_index):
    """Returns function that gives a numpy view of the current tensor buffer.

    This allows reading and writing to this tensors w/o copies. This more
    closely mirrors the C++ Interpreter class interface's tensor() member, hence
    the name. Be careful to not hold these output references through calls
    to `allocate_tensors()` and `invoke()`. This function cannot be used to read
    intermediate results.

    Usage:

    ```
    interpreter.allocate_tensors()
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    for i in range(10):
      input().fill(3.)
      interpreter.invoke()
      print("inference %s" % output())
    ```

    Notice how this function avoids making a numpy array directly. This is
    because it is important to not hold actual numpy views to the data longer
    than necessary. If you do, then the interpreter can no longer be invoked,
    because it is possible the interpreter would resize and invalidate the
    referenced tensors. The NumPy API doesn't allow any mutability of the
    the underlying buffers.

    WRONG:

    ```
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])()
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    interpreter.allocate_tensors()  # This will throw RuntimeError
    for i in range(10):
      input.fill(3.)
      interpreter.invoke()  # this will throw RuntimeError since input,output
    ```

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
        the 'index' field in get_output_details.

    Returns:
      A function that can return a new numpy array pointing to the internal
      TFLite tensor state at any point. It is safe to hold the function forever,
      but it is not safe to hold the numpy array forever.
    """
    return lambda: self._interpreter.tensor(self._interpreter, tensor_index)

  def invoke(self):
    """Invoke the interpreter.

    Be sure to set the input sizes, allocate tensors and fill values before
    calling this. Also, note that this function releases the GIL so heavy
    computation can be done in the background while the Python interpreter
    continues. No other function on this object should be called while the
    invoke() call has not finished.

    Raises:
      ValueError: When the underlying interpreter fails raise ValueError.
    """
    self._ensure_safe()
    self._interpreter.Invoke()

  def reset_all_variables(self):
    return self._interpreter.ResetVariableTensors()

  # Experimental and subject to change.
  def _native_handle(self):
    """Returns a pointer to the underlying tflite::Interpreter instance.

    This allows extending tflite.Interpreter's functionality in a custom C++
    function. Consider how that may work in a custom pybind wrapper:

      m.def("SomeNewFeature", ([](py::object handle) {
        auto* interpreter =
          reinterpret_cast<tflite::Interpreter*>(handle.cast<intptr_t>());
        ...
      }))

    and corresponding Python call:

      SomeNewFeature(interpreter.native_handle())

    Note: This approach is fragile. Users must guarantee the C++ extension build
    is consistent with the tflite.Interpreter's underlying C++ build.
    """
    return self._interpreter.interpreter()


