#ifndef __CONV_H__
#define __CONV_H__

#include <all.h>

int8_t make_conv_TA(layer_TA *layerta,
                    INT_TA channel,
                    INT_TA num,
                    INT_TA size[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA stride[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA padding[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA dilation[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA groups,
                    int8_t isBias,
                    PADDING_MODE_TA padding_mode);
#if CHECK           
void printConv(const layer_TA * const layerta);
#endif

// ==============================================================

void forward_conv2d_TA(base_layer_TA *l, network_TA *net);

#endif