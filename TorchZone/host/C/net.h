#ifndef __NET_H__
#define __NET_H__

#include <head.h>

void make_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                     INT_TA num, INT_TA batch,
                     INT_TA inputs, INT_TA outputs,
                     INT_TA inoutSize,
                     INT_TA workspace_size,
                     FLOAT64_TA clip,
                     int8_t train);

int8_t forward_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                          FLOATCA *input, INT_TA length, 
                          int32_t idx[/* MAX_LAYERS_SEQUENCE_TA */]);
int8_t forward_ret_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                              FLOATCA *output, INT_TA length);

int8_t forwardFetch_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                               FLOATCA *input, INT_TA length_in, 
                               FLOATCA *output, INT_TA length_out, 
                               int32_t idx[/* MAX_LAYERS_SEQUENCE_TA */]);


#endif