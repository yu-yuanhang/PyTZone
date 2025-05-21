#ifndef __TSOPS_H__
#define __TSOPS_H__

#include <all.h>

int8_t make_tsops_TA(layer_TA *layerta,
                     STATION_TA stn,
                     INT_TA index2, int8_t kpIdx2,
                     INT_TA index1, int8_t kpIdx1);

void forward_tsops_TA(base_layer_TA *l, network_TA *net);

#if CHECK           
void printTsops(const layer_TA * const layerta);
#endif

#endif