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



inline
Mat::Mat(int _rows, int _cols, int _type, void* _data)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_rows), cols(_cols),
      data((uchar*)_data), size(&rows), datastart((uchar*)_data)
{
    size_t esz = CV_ELEM_SIZE(_type);
    size_t minstep = cols * esz;
    size_t _step = minstep;

    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step * rows;
    dataend = datalimit - _step + minstep;


    flags |= CONTINUOUS_FLAG;
}

inline
Mat::Mat(const Mat& m, const Rect& roi)
    : flags(m.flags), dims(2), rows(roi.height), cols(roi.width),
    data(m.data + roi.y*m.step[0]), size(&rows),
    datastart(m.datastart), datalimit(m.datalimit), dataend(m.dataend)
{
    size_t esz = CV_ELEM_SIZE(flags);
    data += roi.x*esz;
    if( roi.width < m.cols || roi.height < m.rows )
        flags |= SUBMATRIX_FLAG;

    step[0] = m.step[0]; step[1] = esz;
}

inline
Mat Mat::operator()( const Rect& roi ) const
{
    return Mat(*this, roi);
}

inline
int Mat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int Mat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int Mat::channels() const
{
    return CV_MAT_CN(flags);
}


inline
uchar* Mat::ptr(int y)
{
    return data + step.p[0] * y;
}

template<typename _Tp> inline
_Tp* Mat::ptr(int y)
{
    return (_Tp*)(data + step.p[0] * y);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int y) const
{
    return (const _Tp*)(data + step.p[0] * y);
}

///////////////////////////// MatSize ////////////////////////////

inline
MatSize::MatSize(int* _p)
    : p(_p) {}

inline
int MatSize::dims() const
{
    return (p - 1)[0];
}

inline
Size MatSize::operator()() const
{
    return Size(p[1], p[0]);
}

inline
const int& MatSize::operator[](int i) const
{
    return p[i];
}

inline
int& MatSize::operator[](int i)
{
    return p[i];
}

inline
MatSize::operator const int*() const
{
    return p;
}

inline
bool MatSize::operator == (const MatSize& sz) const
{
    int d = dims();
    int dsz = sz.dims();
    if( d != dsz )
        return false;
    if( d == 2 )
        return p[0] == sz.p[0] && p[1] == sz.p[1];

    for( int i = 0; i < d; i++ )
        if( p[i] != sz.p[i] )
            return false;
    return true;
}

inline
bool MatSize::operator != (const MatSize& sz) const
{
    return !(*this == sz);
}



///////////////////////////// MatStep ////////////////////////////

inline
MatStep::MatStep()
{
    p = buf; p[0] = p[1] = 0;
}

inline
MatStep::MatStep(size_t s)
{
    p = buf; p[0] = s; p[1] = 0;
}

inline
const size_t& MatStep::operator[](int i) const
{
    return p[i];
}

inline
size_t& MatStep::operator[](int i)
{
    return p[i];
}

inline MatStep::operator size_t() const
{
    return buf[0];
}

inline MatStep& MatStep::operator = (size_t s)
{
    buf[0] = s;
    return *this;
}
