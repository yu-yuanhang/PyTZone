#include "tsops.h"

int8_t make_tsops_TA(layer_TA *layerta,
                     STATION_TA stn,
                     INT_TA index2, int8_t kpIdx2,
                     INT_TA index1, int8_t kpIdx1) {
    DMSG("make_tsops_TA has been called");

    layerta->_forward_ta = forward_tsops_TA;

    layerta->_stn = stn;
    layerta->_index2 = index2;
    layerta->_kpIdx2 = kpIdx2;
    layerta->_index1 = index1;
    layerta->_kpIdx1 = kpIdx1;
    
    return 0;
}

void forward_tsops_TA(base_layer_TA *l, network_TA *net) {
    printf("forward_tsops_TA(base_layer_TA *, network_TA *)\n");

    INT_TA outputs = net->_tmp_outputs;
    
    layer_TA *l2 = NULL;
    float *dt2 = NULL;
    if (l->_index2 >= 0) {
        l2 = &(net->_layers[l->_index2]);
        dt2 = (l->_kpIdx2) ? l2->_input : l2->_output;
    } else {dt2 = net->_input;}
    // layer_TA *l1 = NULL;
    // float *dt1 = NULL;
    // if (l->_index1 >= 0) {
    //     l1 = &(net->_layers[l->_index1]);
    //     dt1 = (l->_kpIdx1) ? l1->_input : l1->_output;
    // } else {dt1 = net->_input;}
    float *dt1 = net->_input;

    if (ADD_STATION == l->_stn) {
        for(INT_TA i = 0; i < outputs; ++i)
            net->_output[i] = dt1[i] + dt2[i];
    } else if (NONE_STATION == l->_stn) {
        // 理论上即使什么都不做也最好把 input 拷贝给 output
        memcpy(net->_output, net->_input, outputs * FLOAT_TA_SIZE);
    }
    dataKeep(l, net);
}

#if CHECK           
void printTsops(const layer_TA * const layerta) {
    printf("~~~~~~~~~~~~~ for tsops ~~~~~~~~~~~~~\n");
    printf("activ == %u\n", layerta->_stn);
    printf("index2 == %d\nindex1 == %d\n", 
            layerta->_index2, layerta->_index1);
}
#endif