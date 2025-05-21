#include "ops.h"

// alloc_shareMem(sizeof(float) * 1024);

// // buffer[0] = 9.0;
// // buffer[1] = 8.0;
// // buffer[15] = 5.0;
// *((float *)sm.buffer) = 9.0;
// *((float *)sm.buffer + 1) = 8.0;
// *((float *)sm.buffer + 15) = 5.0;

// printBytes(sm.buffer, 64);

// // wp.buffer = (void *)buffer;
// // printBytes(wp.buffer, 64);
// memset(&op, 0, sizeof(op));
// op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INOUT, TEEC_VALUE_INPUT,
//                     TEEC_NONE, TEEC_NONE);

// op.params[0].memref.parent = &sm;
// op.params[0].memref.offset = 0;
// op.params[0].memref.size = sizeof(float) * 1024;

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
                   FLOATCA *weights, FLOATCA *biases) {

    printf("make_layer_ca(...)\n");

    // 这里 share Mem 的安全性需要进一步封装
    // if (!sm.buffer || sm.size < MAKE_LAYER_LENGTH) {
    //     // EMSG("malloc for buffer error : size = %d", MAKE_NETWORK_LENGTH);
    //     // printf("share memory allocation error : make_network_ca(...) : sm.size = %d\n", sm.size);
    //     errx(1, "share memory allocation error : make_layer_ca(...) : sm.size = %d",
    //          sm.size);
    // }

    void *buffer = (void *)malloc(MAKE_LAYER_LENGTH);

    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    uint32_t offset = 0;
    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &dimsNum, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    PACK(buffer, offset, dims, arrlen);offset += arrlen;
    PACK(buffer, offset, out_dims, arrlen);offset += arrlen;
    
    PACK(buffer, offset, &binary, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &xnor, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &keepIn, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &keepOut, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &batch, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &inptus, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &outputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &nweights, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &nbiases, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &workspace_size, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;
    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    

    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_LAYER_LENGTH;

    op.params[1].tmpref.buffer = weights;
    op.params[1].tmpref.size = nweights * FLOAT_TA_SIZE;
    op.params[2].tmpref.buffer = biases;
    op.params[2].tmpref.size = nbiases * FLOAT_TA_SIZE;

    // uint32_t size = 0;
    // FLOATCA *buffer = NULL;
    // if(BATCHNORM_TYPE == type) {
    //     size = nweights * FLOAT_TA_SIZE;
    //     buffer = (FLOATCA *)malloc(size * 2);
    //     memcpy(buffer, mean, size);
    //     memcpy(buffer + size, variance, size);
    // }
    // op.params[3].tmpref.buffer = buffer;
    // op.params[3].tmpref.size = size * 2;

    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER) failed 0x%x origin 0x%x",
         res, err_origin);
     
}


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
                  uint32_t padding_mode) {
    printf("make_conv_ca(...)\n");

    // 这里 share Mem 的安全性需要进一步封装
    // if (!sm.buffer || sm.size < MAKE_CONV_LENGTH) {
    //     // EMSG("malloc for buffer error : size = %d", MAKE_NETWORK_LENGTH);
    //     // printf("share memory allocation error : make_network_ca(...) : sm.size = %d\n", sm.size);
    //     errx(1, "share memory allocation error : make_conv_ca(...) : sm.size = %d",
    //          sm.size);
    // }
    
    void *buffer = (void *)malloc(MAKE_CONV_LENGTH);

    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    uint32_t offset = 0;
    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;
    PACK(buffer, offset, &channel, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &num, INT_TA_SIZE);offset += INT_TA_SIZE;

    PACK(buffer, offset, size, arrlen);offset += arrlen;
    PACK(buffer, offset, stride, arrlen);offset += arrlen;
    PACK(buffer, offset, padding, arrlen);offset += arrlen;
    PACK(buffer, offset, dilation, arrlen);offset += arrlen;

    PACK(buffer, offset, &groups, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &isBias, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &padding_mode, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_CONV_LENGTH;
    
    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_EXT_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER_EXT) failed 0x%x origin 0x%x",
         res, err_origin);
    
}

void make_norm_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                  uint32_t type, int32_t idx,
                  INT_TA in_c,
                  FLOAT64_TA eps,
                  FLOAT64_TA momentum,
                  int8_t affine,
                  int8_t track_running_stats,
                  FLOATCA *mean, FLOATCA *variance,
                  INT_TA means, INT_TA variances) {
    printf("make_norm_ca(...)\n");

    // if (!sm.buffer || sm.size < MAKE_NORM_LENGTH) {
    //     errx(1, "share memory allocation error : make_norm_ca(...) : sm.size = %d",
    //          sm.size);
    // }

    void *buffer = (void *)malloc(MAKE_NORM_LENGTH);

    uint32_t offset = 0;
    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;
    PACK(buffer, offset, &in_c, INT_TA_SIZE);offset += INT_TA_SIZE;

    PACK(buffer, offset, &eps, FLOAT64_TA_SIZE);offset += FLOAT64_TA_SIZE;
    PACK(buffer, offset, &momentum, FLOAT64_TA_SIZE);offset += FLOAT64_TA_SIZE;

    PACK(buffer, offset, &affine, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &track_running_stats, INT8_T_SIZE);offset += UINT32_T_SIZE;

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_NORM_LENGTH;

    op.params[1].tmpref.buffer = mean;
    op.params[1].tmpref.size = means * FLOAT_TA_SIZE;
    op.params[2].tmpref.buffer = variance;
    op.params[2].tmpref.size = variances * FLOAT_TA_SIZE;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    
    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_EXT_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER_EXT) failed 0x%x origin 0x%x",
         res, err_origin);


}


void make_activ_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                   uint32_t type, int32_t idx, uint32_t activ) {

    printf("make_activ_ca(...)\n");

    // if (!sm.buffer || sm.size < MAKE_ACTIV_LENGTH) {
    //     errx(1, "share memory allocation error : make_activ_ca(...) : sm.size = %d",
    //          sm.size);
    // }
    
    void *buffer = (void *)malloc(MAKE_ACTIV_LENGTH);
    
    uint32_t offset = 0;
    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;
    PACK(buffer, offset, &activ, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_ACTIV_LENGTH;

    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_EXT_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER_EXT) failed 0x%x origin 0x%x",
         res, err_origin);

}

void make_tsops_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                   uint32_t type, int32_t idx, 
                   uint32_t stn, 
                   INT_TA index2, int8_t kpIdx2,
                   INT_TA index1, int8_t kpIdx1) {
    printf("make_tsops_ca(...)\n");
    // if (!sm.buffer || sm.size < MAKE_STATION_LENGTH) {
    //     errx(1, "share memory allocation error : make_activ_ca(...) : sm.size = %d",
    //          sm.size);
    // }

    void *buffer = (void *)malloc(MAKE_STATION_LENGTH);

    uint32_t offset = 0;
    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;
    PACK(buffer, offset, &stn, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &index2, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &kpIdx2, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &index1, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &kpIdx1, INT8_T_SIZE);offset += UINT32_T_SIZE;

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_STATION_LENGTH;
    
    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_EXT_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER_EXT) failed 0x%x origin 0x%x",
         res, err_origin);
}

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
                  uint32_t padding_mode) {
    printf("make_pool_ca(...)\n");

    // if (!sm.buffer || sm.size < MAKE_POOL_LENGTH) {
    //     errx(1, "share memory allocation error : make_pool_ca(...) : sm.size = %d",
    //          sm.size);
    // }

    void *buffer = (void *)malloc(MAKE_POOL_LENGTH);
    
    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    uint32_t offset = 0;

    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;

    PACK(buffer, offset, size, arrlen);offset += arrlen;
    PACK(buffer, offset, stride, arrlen);offset += arrlen;
    PACK(buffer, offset, padding, arrlen);offset += arrlen;
    PACK(buffer, offset, dilation, arrlen);offset += arrlen;

    PACK(buffer, offset, &return_indices, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &ceil_mode, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &count_include_pad, INT8_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &divisor_override, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &padding_mode, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_POOL_LENGTH;
    
    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_EXT_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER_EXT) failed 0x%x origin 0x%x",
         res, err_origin);

}

void make_linear_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                    uint32_t type, int32_t idx,
                    INT_TA in_c,
                    INT_TA out_c,
                    int8_t isBias) {
    printf("make_linear_ca(...)\n");

    // if (!sm.buffer || sm.size < MAKE_FCONNECTED_LENGTH) {
    //     errx(1, "share memory allocation error : make_linear_ca(...) : sm.size = %d",
    //          sm.size);
    // }

    void *buffer = (void *)malloc(MAKE_FCONNECTED_LENGTH);

    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    uint32_t offset = 0;

    PACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    PACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;

    PACK(buffer, offset, &in_c, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &out_c, INT_TA_SIZE);offset += INT_TA_SIZE;

    PACK(buffer, offset, &isBias, INT8_T_SIZE);offset += UINT32_T_SIZE;


    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,  
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE);
    
    op.params[0].tmpref.buffer = buffer;
    op.params[0].tmpref.size = MAKE_FCONNECTED_LENGTH;
    
    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_LAYER_EXT_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_LAYER_EXT) failed 0x%x origin 0x%x",
         res, err_origin);
}