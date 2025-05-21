#ifndef __IM2COL_TA_H__
#define __IM2COL_TA_H__

#include <all.h>

void im2col_cpu_TA(float* data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_col);
void im2col_cpu_TA_2d(float* data_im,
                int channels,  int height,  int width,
                int ksize_h, int ksize_w, 
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                float* data_col);

float im2col_get_pixel_TA(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad);
float im2col_get_pixel_TA_2d(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad_h, int pad_w);

#endif
