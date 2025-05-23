#include "im2col_TA.h"
#include <stdio.h>

float im2col_get_pixel_TA(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_TA(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    printf("im2col_cpu_TA(...)\n");
    // 关于卷积核不是矩形的情况后续再补 ......
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    // channels_col 表示输入特征图展开后的行数
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_TA(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}


// ==================================================================

float im2col_get_pixel_TA_2d(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad_h, int pad_w)
{
    row -= pad_h;
    col -= pad_w;
    
    // 这里暂时都默认 padding mode 为 zero
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu_TA_2d(float* data_im,
                int channels,  int height,  int width,
                int ksize_h, int ksize_w, 
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad_h - ksize_h) / stride_h + 1;
    int width_col = (width + 2*pad_w - ksize_w) / stride_w + 1;
    // channels_col 表示输入特征图展开后的行数
    int channels_col = channels * ksize_h * ksize_w;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im = c / ksize_w / ksize_h;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride_h;
                int im_col = w_offset + w * stride_w;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_TA_2d(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad_h, pad_w);
            }
        }
    }
}

