#ifndef __POOL_H__
#define __POOL_H__

#include <all.h>

int8_t make_pool_TA(layer_TA *layerta,
                    INT_TA size[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA stride[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA padding[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA dilation[/* MAX_CONV_DIMENSIONS_TA */],
                    int8_t return_indices,
                    int8_t ceil_mode,
                    int8_t count_include_pad,
                    INT_TA divisor_override,
                    PADDING_MODE_TA padding_mode);

void forward_maxpool2d_TA(base_layer_TA *l, network_TA *net);
void forward_avgpool2d_TA(base_layer_TA *l, network_TA *net);

#if CHECK           
void printPool(const layer_TA * const layerta);
#endif

#endif