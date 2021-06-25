Image Transform
==========================


This does a perspective transform on an image.
This code was copied and slightly modified from OpenCV.

See the following for more details on a perspective transform:
https://www.geeksforgeeks.org/perspective-transformation-python-opencv/




The basic usage is:

```c
#include "image_transform.h"

const float src_points[8] = 
{
    /* top-left x, top-left y, */
    /* top-right x, top-right y, */
    /* bottom-right x, bottom-right y, */
    /* bottom-left x, bottom-left y */
};
const float dst_points[8] = {
    /* top-left x, top-left y, */
    /* top-right x, top-right y, */
    /* bottom-right x, bottom-right y, */
    /* bottom-left x, bottom-left y */
};

float warp[9];

// Calculate the transform matrix
image_transform_get_transform(src_points, dst_points, warp);


const uint8_t src_img[src_height][src_width][channels];
uint8_t dst_img[dst_height][dst_width][channels];

// Apply a perspective transform to the src_img
// and fill the transformed dst_img
image_transform(
    src_img, src_width, src_height,
    dst_img, dst_width, dst_width,
    channels, warp
)


```
