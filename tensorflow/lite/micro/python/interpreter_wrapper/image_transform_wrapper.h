
#include <vector>
#include <Python.h>


namespace tflite {
namespace image_transform {

PyObject* GetPerspectiveTransformMatrix(
    const std::vector<float>& src_points, 
    const std::vector<float>& dst_points
);

PyObject* ApplyPerspectiveTransform(
    PyObject* img, 
    int dst_width,
    int dst_height,
    PyObject* perspective_transform_matrix,
    bool standardize
);

} // namespace image_transform 
} // namespace tflite

