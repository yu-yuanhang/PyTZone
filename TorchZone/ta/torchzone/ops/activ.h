#ifndef __ACTIV_H__
#define __ACTIV_H__

#include <all.h>

int8_t make_activ_TA(layer_TA *layerta, ACTIVATION_TA activ);

void activate_array_TA(base_layer_TA *l, network_TA *net);
FLOATTA (*getActivate_TA(ACTIVATION_TA activ)) (FLOATTA);

#if CHECK           
void printActiv(const layer_TA * const layerta);
#endif

static inline float logistic_activate_TA(float x){return 1./(1. + ta_exp(-x));}
static inline float relu_activate_TA(float x){return x*(x>0);}
static inline float relie_activate_TA(float x){return (x>0) ? x : .01*x;}
static inline float linear_activate_TA(float x){return x;}
static inline float ramp_activate_TA(float x){return x*(x>0)+.1*x;}
static inline float tanh_activate_TA(float x){return (ta_exp(2*x)-1)/(ta_exp(2*x)+1);}
static inline float plse_activate_TA(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}
static inline float leaky_activate_TA(float x){return (x>0) ? x : (1e-2)*x;}
static inline float elu_activate_TA(float x){return (x >= 0)*x + (x < 0)*(ta_exp(x)-1);}
static inline float loggy_activate_TA(float x){return 2./(1. + ta_exp(-x)) - 1;}
static inline float stair_activate_TA(float x)
{
    int n = ta_floor(x);
    if (n%2 == 0) return ta_floor(x/2.);
    else return (x - n) + ta_floor(x/2.);
}
static inline float hardtan_activate_TA(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float lhtan_activate_TA(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float selu_activate_TA(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(ta_exp(x)-1);}

#endif