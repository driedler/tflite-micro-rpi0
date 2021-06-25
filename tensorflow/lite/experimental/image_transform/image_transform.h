
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


// Generate perspective transform matrix
void image_transform_get_transform(const float *src_points, const float *dst_points, float m[9]);

// Apply perspective transform to src and populate dst
void image_transform_invoke(
    const uint8_t *src_data, int src_width, int src_height,
    uint8_t *dst_data, int dst_width, int dst_height,
    int channels, const float m[9]
);

// dst = (src - mean(src)) / std(src)
void standardize_mean_and_std(const uint8_t *src, float *dst, int length);



#ifdef __cplusplus
}
#endif