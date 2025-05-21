#ifndef __OPS_H__
#define __OPS_H__

#include <C/net.h>
#include <head.h>

// INT_TA batch,
// INT_TA inptus, INT_TA outputs,
// INT_TA nweights, INT_TA nbiases,
// INT_TA workspace_size,
void make_layer_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                   uint32_t type,
                   uint32_t dimsNum,
                   INT_TA dims[/* MAX_CONV_DIMENSIONS_TA */],
                   INT_TA out_dims[/* MAX_CONV_DIMENSIONS_TA */],
                   int8_t binary, int8_t xnor,
                   int8_t keepIn, int8_t keepOut,
                   INT_TA batch,
                   INT_TA inptus, INT_TA outputs,
                   INT_TA nweights, INT_TA nbiases,
                   INT_TA workspace_size,
                   int32_t idx,
                   FLOATCA *weights, FLOATCA *biases);


void make_conv_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                  uint32_t type, int32_t idx, // type 和 idx 是必要的
                  INT_TA channel,
                  INT_TA num,
                  INT_TA size[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA stride[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA padding[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA dilation[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA groups,
                  int8_t isBias,
                  uint32_t padding_mode);

void make_norm_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                  uint32_t type, int32_t idx,
                  INT_TA in_c,
                  FLOAT64_TA eps,
                  FLOAT64_TA momentum,
                  int8_t affine,
                  int8_t track_running_stats,
                  FLOATCA *mean, FLOATCA *variance,
                  INT_TA means, INT_TA variances);

void make_activ_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                   uint32_t type, int32_t idx, uint32_t activ);

void make_tsops_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                   uint32_t type, int32_t idx, 
                   uint32_t stn, 
                   INT_TA index2, int8_t kpIdx2, 
                   INT_TA index1, int8_t kpIdx1);

void make_pool_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                  uint32_t type, int32_t idx,
                  INT_TA size[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA stride[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA padding[/* MAX_CONV_DIMENSIONS_TA */],
                  INT_TA dilation[/* MAX_CONV_DIMENSIONS_TA */],
                  int8_t return_indices,
                  int8_t ceil_mode,
                  int8_t count_include_pad,
                  INT_TA divisor_override,
                  uint32_t padding_mode);

void make_linear_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                    uint32_t type, int32_t idx,
                    INT_TA in_c,
                    INT_TA out_c,
                    int8_t isBias);



#endif