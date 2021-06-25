#include <functional>
#include <memory>

#include "tensorflow/lite/micro/python/interpreter_wrapper/image_transform_wrapper.h"
#include "tensorflow/lite/experimental/image_transform/image_transform.h"
#include "tensorflow/lite/micro/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/micro/python/interpreter_wrapper/python_utils.h"


namespace tflite {
namespace image_transform {


using python_utils::PyDecrefDeleter;



PyObject* PyArrayFromFloatVector(const float* data, npy_intp size) {
  void* pydata = malloc(size * sizeof(float));
  memcpy(pydata, data, size * sizeof(float));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_FLOAT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}


PyObject* GetPerspectiveTransformMatrix(
    const std::vector<float>& src_points, 
    const std::vector<float>& dst_points
)
{
    float perspective_transform_matrix[9];
    
    python::ImportNumpy();

    image_transform_get_transform(
        src_points.data(), dst_points.data(), perspective_transform_matrix
    );
    return PyArrayFromFloatVector(perspective_transform_matrix, 9);
}

PyObject* ApplyPerspectiveTransform(
    PyObject* img, 
    int dst_width,
    int dst_height,
    PyObject* perspective_transform_matrix,
    bool standardize
)
{
    std::unique_ptr<PyObject, PyDecrefDeleter> img_safe(
        PyArray_FromAny(img, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
    if (!img_safe) {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to convert img into readable numpy array.");
        return nullptr;
    }

    PyArrayObject* img_array = reinterpret_cast<PyArrayObject*>(img_safe.get());
    if(PyArray_NDIM(img_array) != 3 || PyArray_TYPE(img_array) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError,
                        "img must be uint8 and 3dims");
        return nullptr;
    }

    if(dst_width <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid dst_width");
        return nullptr;
    }

    if(dst_height <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid dst_height");
        return nullptr;
    }

    std::unique_ptr<PyObject, PyDecrefDeleter> warp_safe(
        PyArray_FromAny(perspective_transform_matrix, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
    if (!warp_safe) {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to convert perspective_transform_matrix into readable numpy array.");
        return nullptr;
    }

    PyArrayObject* warp_array = reinterpret_cast<PyArrayObject*>(warp_safe.get());
    if(PyArray_NDIM(warp_array) != 1 || PyArray_SHAPE(warp_array)[0] != 9 || PyArray_TYPE(warp_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_ValueError,
                        "perspective_transform_matrix must be float32 and contain 9 elements");
        return nullptr;
    }

    auto src = static_cast<const uint8_t*>(PyArray_DATA(img_array));
    const int src_height = PyArray_SHAPE(img_array)[0];
    const int src_width = PyArray_SHAPE(img_array)[1];
    const int src_depth = PyArray_SHAPE(img_array)[2];
    auto warp = static_cast<const float*>(PyArray_DATA(warp_array));


    const int dst_length = dst_width*dst_height*src_depth;
    const int element_size = standardize ? sizeof(float) : sizeof(uint8_t);

    auto dst = static_cast<uint8_t*>(malloc(dst_length));
    if(dst == nullptr) {
        PyErr_SetString(PyExc_ValueError,
                        "failed to alloc dst buffer");
        return nullptr;
    }

    image_transform_invoke(src, src_width, src_height, dst, dst_width, dst_height, src_depth, warp);

    npy_intp dst_dims[3] = { dst_height, dst_width, src_depth };
    PyObject* dst_obj;

    if(standardize)
    {
        auto standard_dst = static_cast<float*>(malloc(dst_length*sizeof(float)));
        if(standard_dst == nullptr) {
            free(dst);
            PyErr_SetString(PyExc_ValueError,
                            "failed to alloc stand_dst buffer");
            return nullptr;
        }
        standardize_mean_and_std(dst, standard_dst, dst_length);
        free(dst);
        dst_obj = PyArray_SimpleNewFromData(3, dst_dims, NPY_FLOAT32, standard_dst);
    }
    else 
    {
       dst_obj = PyArray_SimpleNewFromData(3, dst_dims, NPY_UINT8, dst);
    }

    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(dst_obj), NPY_ARRAY_OWNDATA);

    return dst_obj;
}

} // namespace image_transform 
} // namespace tflite
