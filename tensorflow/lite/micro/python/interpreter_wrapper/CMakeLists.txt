
if(NOT DEFINED PYTHON)
set(PYTHON python3.7)
endif()


execute_process(COMMAND ${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON} -c "import pybind11; print (pybind11.get_include())" OUTPUT_VARIABLE PYBIND11_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON} -c "import numpy; print (numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "PYTHON_INCLUDE=${PYTHON_INCLUDE}")
message(STATUS "PYBIND11_INCLUDE=${PYBIND11_INCLUDE}")
message(STATUS "NUMPY_INCLUDE=${NUMPY_INCLUDE}")


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../experimental/image_transform ${CMAKE_CURRENT_BINARY_DIR}/image_transform)


add_library(_pywrap_tensorflow_interpreter_wrapper SHARED EXCLUDE_FROM_ALL)
target_sources(_pywrap_tensorflow_interpreter_wrapper
PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/interpreter_wrapper.cc
  ${CMAKE_CURRENT_LIST_DIR}/interpreter_wrapper_pybind11.cc
  ${CMAKE_CURRENT_LIST_DIR}/numpy.cc
  ${CMAKE_CURRENT_LIST_DIR}/allocation.cc
  ${CMAKE_CURRENT_LIST_DIR}/model_builder.cc
  ${CMAKE_CURRENT_LIST_DIR}/python_error_reporter.cc
  ${CMAKE_CURRENT_LIST_DIR}/python_utils.cc
  ${CMAKE_CURRENT_LIST_DIR}/image_transform_wrapper.cc 
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
  image_transform
)

