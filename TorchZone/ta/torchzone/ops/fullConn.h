#ifndef __FULLCONN_H__
#define __FULLCONN_H__

#include <all.h>

int8_t make_linear_TA(layer_TA *layerta,
                      INT_TA in_c,
                      INT_TA out_c,
                      int8_t isBias);

void forward_linear_TA(base_layer_TA *l, network_TA *net);

#if CHECK           
void printLinear(const layer_TA * const layerta);
#endif

#endif