#ifndef __NORM_H__
#define __NORM_H__

#include <all.h>

int8_t make_norm_TA(layer_TA *layerta,
                    INT_TA in_c,
                    FLOAT64_TA eps,
                    FLOAT64_TA momentum,
                    int8_t affine,
                    int8_t track_running_stats,
                    FLOATCA *mean, FLOATCA *variance,
                    uint32_t means, uint32_t variances);


void forward_batchnorm2d_TA(base_layer_TA *l, network_TA *net);

void normalize_cpu_TA(float *x, float *mean, float *variance, int batch, int filters, int spatial);

#if CHECK           
void printNorm(const layer_TA * const layerta);
#endif

#endif