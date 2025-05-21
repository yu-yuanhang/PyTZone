#include "activ.h"

int8_t make_activ_TA(layer_TA *layerta, ACTIVATION_TA activ) {
    DMSG("make_activ_TA has been called");
    // 查找对应的激活函数 
    // layerta->_forward_ta = forward_conv2d_TA;
    layerta->_forward_ta = activate_array_TA;
    layerta->_activ = activ;
    layerta->_activate_TA = getActivate_TA(activ);

    return 0;
}

void activate_array_TA(base_layer_TA *l, network_TA *net) {
    // FLOATTA (*_activate_TA) (FLOATTA);
    // _activate_TA = getActivate_TA(l->_activ);
    uint32_t num = l->_outputs * l->_batch;
    int i;
    for (i = 0; i < num; ++i) {net->_output[i] = l->_activate_TA(net->_input[i]);}
}

FLOATTA (*getActivate_TA(ACTIVATION_TA activ)) (FLOATTA)
{
    switch(activ){
        case LOGISTIC_ACTIV:
            return logistic_activate_TA;
        case RELU_ACTIV:
            return relu_activate_TA;
        case RELIE_ACTIV:
            return relie_activate_TA;
        case LINEAR_ACTIV:
            return linear_activate_TA;
        case RAMP_ACTIV:
            return ramp_activate_TA;
        case TANH_ACTIV:
            return tanh_activate_TA;
        case PLSE_ACTIV:
            return plse_activate_TA;
        case LEAKY_ACTIV:
            return leaky_activate_TA;
        case ELU_ACTIV:
            return elu_activate_TA;
        case LOGGY_ACTIV:
            return loggy_activate_TA;
        case STAIR_ACTIV:
            return stair_activate_TA;
        case HARDTAN_ACTIV:
            return hardtan_activate_TA;
        case LHTAN_ACTIV:
            return lhtan_activate_TA;
        case SELU_ACTIV:
            return selu_activate_TA;
    }
    return 0;
}

#if CHECK           
void printActiv(const layer_TA * const layerta) {
    printf("~~~~~~~~~~~~~ for activ ~~~~~~~~~~~~~\n");
    printf("activ == %u\n", layerta->_activ);
}
#endif
