/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/



#include <algorithm>
#include <cstdio>

#include "internal/types.hpp"
#include "internal/mat.hpp"
#include "internal/saturate.hpp"






static void remap(Mat &src, Mat &dst, Mat &map1);
template<typename T>
static void remapNearest( const Mat& _src, Mat& _dst, const Mat& _xy);


static const int BLOCK_SZ = 32;
static short XY[BLOCK_SZ*BLOCK_SZ*2];




extern "C"  void image_transform_invoke(
    const uint8_t *src_data, int src_width, int src_height,
    uint8_t *dst_data, int dst_width, int dst_height,
    int channels, const float m[9]
)
{
    const int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    Mat src(src_height, src_width, type, (uint8_t*)src_data);
    Mat dst(dst_height, dst_width, type, dst_data);


    int bh0 = std::min(BLOCK_SZ/2, dst_height);
    int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, dst_width);
    bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, dst_height);


    for(int y = 0; y < dst_height; y += bh0 )
    {
        for(int x = 0; x < dst_width; x += bw0 )
        {
            int bw = std::min( bw0, dst_width - x);
            int bh = std::min( bh0, dst_height - y); // height

            Mat _XY(bh, bw, CV_16SC2, XY);
            Mat dpart(dst, Rect(x, y, bw, bh));

            for(int y1 = 0; y1 < bh; y1++ )
            {
                short* xy = XY + y1*bw*2;
                float X0 = m[0]*x + m[1]*(y + y1) + m[2];
                float Y0 = m[3]*x + m[4]*(y + y1) + m[5];
                float W0 = m[6]*x + m[7]*(y + y1) + m[8];


                for( int x1 = 0; x1 < bw; x1++ )
                {
                    float W = W0 + m[6]*x1;
                    W = W ? 1.f/W : 0;
                    float fX = std::max((float)INT_MIN, std::min((float)INT_MAX, (X0 + m[0]*x1)*W));
                    float fY = std::max((float)INT_MIN, std::min((float)INT_MAX, (Y0 + m[3]*x1)*W));
                    int X = saturate_cast<int>(fX);
                    int Y = saturate_cast<int>(fY);

                    xy[x1*2] = saturate_cast<short>(X);
                    xy[x1*2+1] = saturate_cast<short>(Y);
                }

            }

            remap( src, dpart, _XY);
        }
    }
}




/*************************************************************************************************/
static void remap(Mat &src, Mat &dst, Mat &map1)
{
    const int buf_size = 1 << 9;
    int brows0 = std::min(16, dst.rows);
    int bcols0 = std::min(buf_size/brows0, dst.cols);
    brows0 = std::min(buf_size/bcols0, dst.rows);


    for(int y = 0; y < dst.rows; y += brows0 )
    {
        for(int x = 0; x < dst.cols; x += bcols0 )
        {
            int brows = std::min(brows0, dst.rows - y);
            int bcols = std::min(bcols0, dst.cols - x);
            Mat dpart(dst, Rect(x, y, bcols, brows));
            Mat bufxy = map1(Rect(x, y, bcols, brows));

            remapNearest<uchar>(src, dpart, bufxy);
        }
    }
}


/*************************************************************************************************/
static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}


/*************************************************************************************************/
template<typename T>
static void remapNearest( const Mat& _src, Mat& _dst, const Mat& _xy)
{
    Size ssize = _src.size(), dsize = _dst.size();
    const int cn = _src.channels();
    const T* S0 = _src.ptr<T>();
    size_t sstep = _src.step/sizeof(S0[0]);

    const unsigned width1 = ssize.width, height1 = ssize.height;


    for(int dy = 0; dy < dsize.height; dy++ )
    {
        T* D = _dst.ptr<T>(dy);
        const short* XY = _xy.ptr<short>(dy);

        for(int dx = 0; dx < dsize.width; dx++, D += cn )
        {
            int sx = XY[dx*2], sy = XY[dx*2+1];
            const T *S;
            if( (unsigned)sx < width1 && (unsigned)sy < height1 )
            {
                if( cn == 3 )
                {
                    S = S0 + sy*sstep + sx*3;
                    D[0] = S[0], D[1] = S[1], D[2] = S[2];
                }
                else if( cn == 4 )
                {
                    S = S0 + sy*sstep + sx*4;
                    D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
                }
                else
                {
                    S = S0 + sy*sstep + sx*cn;
                    for(int k = 0; k < cn; k++ )
                        D[k] = S[k];
                }
            }
            else
            {
                sx = clip(sx, 0, ssize.width);
                sy = clip(sy, 0, ssize.height);
                S = S0 + sy*sstep + sx*cn;

                for(int k = 0; k < cn; k++ )
                    D[k] = S[k];
            }
        }
    }
}

