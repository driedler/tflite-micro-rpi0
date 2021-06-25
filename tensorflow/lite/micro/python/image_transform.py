from typing import Tuple,List
import numpy as np


try:
  from tflite_micro_runtime import _pywrap_tflm_interpreter_wrapper as _interpreter_wrapper
except:
  import _pywrap_tflm_interpreter_wrapper as _interpreter_wrapper



class ImageTransformer(object):

    def __init__(
        self, 
        src_points: list, 
        dst_size:Tuple[int,int], 
        standardize:bool=False
    ):
        dst_width = dst_size[0]
        dst_height = dst_size[1]

        dst_points = [ 
            0, 0,
            dst_width-1,0,
            dst_width-1,dst_height-1,
            0,dst_height-1
        ]

        if isinstance(src_points[0], (tuple,list)):
            pts = []
            for pt in src_points:
                pts.append(pt[0])
                pts.append(pt[1])
            src_points = pts
        elif isinstance(src_points, np.ndarray):
            src_points = src_points.flatten()

        src_points = [x for x in map(float, src_points)]
        dst_points = [x for x in map(float, dst_points)]

        self._perspective_transform_matrix = _interpreter_wrapper.GetPerspectiveTransformMatrix(
            src_points,
            dst_points
        )
        self._dst_width = dst_width
        self._dst_height = dst_height
        self.standardize = standardize

    def invoke(self, img: np.ndarray) -> np.ndarray:
        return _interpreter_wrapper.ApplyPerspectiveTransform(
            img, 
            self._dst_width,
            self._dst_height,
            self._perspective_transform_matrix,
            self.standardize
        )