
#include <cmath>


#include "image_transform.h"



/*************************************************************************************************/
extern "C" void standardize_mean_and_std(const uint8_t *src, float *dst, int length)
{
    const int src_length = length;
    float mean = 0.0f;
    float count = 0.0f;
    float m2 = 0.0f;

    // Calculate the STD and mean
    const uint8_t *src_ptr = src;
    for(int i = src_length; i > 0; --i)
    {
        const float value = (float)(*src_ptr++);

        count += 1;

        const float delta = value - mean;
        mean += delta / count;
        const float delta2 = value - mean;
        m2 += delta * delta2;
    }

    const float variance = m2 / count;
    const float std = sqrt(variance);
    const float std_recip = 1.0f / std; // multiplication is faster than division

    // Subtract the mean and divide by the STD
    src_ptr = src;
    for(int i = src_length; i > 0; --i)
    {
        const float value = (float)(*src_ptr++);
        const float x = value - mean;

        *dst++ = x * std_recip;
    }
}
