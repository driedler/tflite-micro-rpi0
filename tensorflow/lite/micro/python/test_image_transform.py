import os 
import sys 
import numpy as np
from cv2 import cv2 

curdir = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.normpath(f'{curdir}/../../../../build/')

sys.path.append(curdir)
sys.path.append(build_dir)

try:
    from .image_transform import ImageTransformer
except:
    from image_transform import ImageTransformer

img_path = f'{curdir}/test_data/test.jpg'

img = cv2.imread(img_path)

xfrm = ImageTransformer(
    src_points=[(231, 180), (404, 212), (409, 364), (227, 368)],
    dst_size=(96,96)
)

xfm_img = xfrm.invoke(img)

cv2.imwrite(f'{curdir}/test_data/test_gen.jpg', xfm_img)

xfrm = ImageTransformer(
    src_points=[(231, 180), (404, 212), (409, 364), (227, 368)],
    dst_size=(96,96),
    standardize=True
)

xfm_img = xfrm.invoke(img)
xfm_img = cv2.normalize(xfm_img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)
cv2.imwrite(f'{curdir}/test_data/test_gen_stand.jpg', xfm_img)