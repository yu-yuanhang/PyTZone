#include "torchzone.h"


// ==========================================================================

TEE_Result make_network_ta_demo(uint32_t param_types, TEE_Param params[4]) {

    float *buffer = NULL;

    printf("&netta = %p\n", &netta);

    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
	DMSG("make_network_ta_demo has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    // param0 fot network
    // param1 for Layers
    // ====================================================
    int8_t i = -10;
    printf("i = %d, INT8_T_SIZE = %ld\n", i, sizeof(i));
    printf("sizeof(int32_t) = %ld\n", sizeof(int32_t));
    printf("sizeof(size_t) = %ld\n", sizeof(size_t));
    printf("sizeof(float) = %ld\n", sizeof(float));
    printf("sizeof(ptr) = %ld\n", sizeof(&i));
    printf("sizeof(int) = %ld\n", sizeof(int));
    int64_t j = 20;
    printf("j = %ld, sizeof(int8=64_t) = %ld\n", j, sizeof(j));
    // ====================================================
    printf("make_network_ta(...)\n");

    buffer = (float *)params[0].memref.buffer;
    printBytes(params[0].memref.buffer, 64);

    float *tmp = aligned_malloc(1024 * sizeof(float), ALIGNMENT);
    printBytes(tmp, 64);
    // params[0].memref.buffer = (void *)tmp;

    // free(buffer);

    printf("params[1].value.a; = %u\n", params[1].value.a);


    params[1].value.a = 888;    
	*((float *)params[0].memref.buffer) = 99.0;
	*((float *)params[0].memref.buffer + 1) = 99.0;
	*((float *)params[0].memref.buffer + 15) = 88.0;
    printf("after deal\n");
    printf("params[1].value.a; = %u\n", params[1].value.a);
    printBytes(params[0].memref.buffer, 64);

    float fl = 9.987;
    printf("fi == %f\n", fl);

    uint32_t arr[10];
    printf("sizeof arr = %ld\n", sizeof(arr));

    aligned_free(tmp);
    return TEE_SUCCESS;    
}

// ==========================================================================

// typedef union {
//     struct {
//         void *buffer;
//         size_t size;
//     } memref;
//     struct {
//         uint32_t a;
//         uint32_t b;
//     } value;
// } TEE_Param;

TEE_Result make_network_ta(uint32_t param_types, TEE_Param params[4]) {

    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
	DMSG("make_network_ta has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    // int8_t make_network_TA(INT_TA num, INT_TA batch, 
    //                     INT_TA inputs, INT_TA outputs,
    //                     INT_TA workspace_size,
    //                     FLOAT64_TA clip,
    //                     int8_t train,
    //                     INT_TA index[MAX_LAYERS_SEQUENCE_TA]) 

    unsigned char *buffer = (unsigned char *)params[0].memref.buffer;

    // 这里会报关于内存对其的警告 
    // 但是这个 buffer 是包装过的所以理论上不存在越界的情况
    uint32_t offset = 0;
    INT_TA num;UNPACK(buffer, offset, &num, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA batch;UNPACK(buffer, offset, &batch, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA inputs;UNPACK(buffer, offset, &inputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA outputs;UNPACK(buffer, offset, &outputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA inoutSize;UNPACK(buffer, offset, &inoutSize, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA workspace_size;UNPACK(buffer, offset, &workspace_size, INT_TA_SIZE);offset += INT_TA_SIZE;
    FLOAT64_TA clip;UNPACK(buffer, offset, &clip, FLOAT64_TA_SIZE);offset += FLOAT64_TA_SIZE;
    int8_t train;UNPACK(buffer, offset, &train, INT8_T_SIZE);offset += UINT32_T_SIZE;

    // int8_t *index = (int8_t *)(buffer + 25);

    int8_t ret = make_network_TA(num, batch,
                                 inputs, outputs, 
                                 inoutSize,
                                 workspace_size,
                                 clip, train);

#if CHECK
    DMSG("make_network_ta finish\n");
    printNet(&netta);
#endif

    return TEE_SUCCESS;
}

TEE_Result make_layer_ta(uint32_t param_types, TEE_Param params[4]) {

    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);
	DMSG("make_layer_ta has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    unsigned char *buffer = (unsigned char *)params[0].memref.buffer;
    uint32_t offset = 0;

    uint32_t type;UNPACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    uint32_t dimsNum;UNPACK(buffer, offset, &dimsNum, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    INT_TA dims[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, dims, arrlen);offset += arrlen;
    INT_TA out_dims[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, out_dims, arrlen);offset += arrlen;

    int8_t binary;UNPACK(buffer, offset, &binary, INT8_T_SIZE);offset += UINT32_T_SIZE;
    int8_t xnor;UNPACK(buffer, offset, &xnor, INT8_T_SIZE);offset += UINT32_T_SIZE;
    int8_t keepIn;UNPACK(buffer, offset, &keepIn, INT8_T_SIZE);offset += UINT32_T_SIZE;
    int8_t keepOut;UNPACK(buffer, offset, &keepOut, INT8_T_SIZE);offset += UINT32_T_SIZE;

    INT_TA batch;UNPACK(buffer, offset, &batch, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA inputs;UNPACK(buffer, offset, &inputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA outputs;UNPACK(buffer, offset, &outputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA nweights;UNPACK(buffer, offset, &nweights, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA nbiases;UNPACK(buffer, offset, &nbiases, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA workspace_size;UNPACK(buffer, offset, &workspace_size, INT_TA_SIZE);offset += INT_TA_SIZE;
    int32_t idx;UNPACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;

    FLOATTA *weights = (FLOATTA *)(params[1].memref.buffer);
    FLOATTA *biases = (FLOATTA *)(params[2].memref.buffer);
    
    // FLOATTA *mean = NULL;
    // FLOATTA *variance = NULL;

    // if((uint32_t)BATCHNORM_TYPE == type) {
    //     mean = (FLOATTA *)(params[3].memref.buffer);
    //     variance = (FLOATTA *)(params[3].memref.buffer + (nweights * FLOAT_TA_SIZE));
    // }

    initLayer((netta._layers + idx),
               // network_TA *netta,
               (LAYER_TYPE_TA)type, 
               (DIMENSIONALITY_TA)dimsNum,
               dims, out_dims,
               binary, xnor,
               keepIn, keepOut,
               batch,
               inputs, outputs,
               //  FLOATTA *weights, FLOATTA *biases,
               nweights, nbiases,
               workspace_size,
               idx,
               weights, biases);

#if CHECK
    DMSG("make_layer_ta finish\n");
#endif
    return TEE_SUCCESS;

}

TEE_Result make_layer_ext_ta(uint32_t param_types, TEE_Param params[4]) {

    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);
	DMSG("make_layer_ext_ta has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    unsigned char *buffer = (unsigned char *)params[0].memref.buffer;
    uint32_t offset = 0;
    uint32_t type;UNPACK(buffer, offset, &type, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    int32_t idx;UNPACK(buffer, offset, &idx, INT32_T_SIZE);offset += INT32_T_SIZE;

    FLOATTA *mean = NULL;
    FLOATTA *variance = NULL;
    if((uint32_t)BATCHNORM_TYPE == type) {
        mean = (FLOATTA *)(params[1].memref.buffer);
        variance = (FLOATTA *)(params[2].memref.buffer);
    }

    switch (type) {
        case CONV_TYPE:
            make_conv_ta((buffer + offset), idx);
            break;
        case MAXPOOL_TYPE:
            make_pool_ta((buffer + offset), idx);
            break;
        case FCONNECTED_TYPE:
            make_linear_ta((buffer + offset), idx);
            break;
        case BATCHNORM_TYPE:
            make_norm_ta((buffer + offset), idx, 
                         mean, variance,
                         params[1].memref.size,
                         params[2].memref.size);
            break;
        case ACTIV_TYPE:
            make_activ_ta((buffer + offset), idx);
            break;
        case TSTATION_TYPE:
            make_tsops_ta((buffer + offset), idx);
            break;
        default:
	        break;
    }
#if CHECK
    DMSG("make_layer_ext_ta finish\n");
    printLayer(netta._layers + idx);
#endif
    return TEE_SUCCESS;
}

void make_conv_ta(void *tar, int32_t idx) {

    unsigned char *buffer = (unsigned char *)tar;
    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    uint32_t offset = 0;

    INT_TA channel;UNPACK(buffer, offset, &channel, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA num;UNPACK(buffer, offset, &num, INT_TA_SIZE);offset += INT_TA_SIZE;

    INT_TA size[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, size, arrlen);offset += arrlen;
    INT_TA stride[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, stride, arrlen);offset += arrlen;
    INT_TA padding[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, padding, arrlen);offset += arrlen;
    INT_TA dilation[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, dilation, arrlen);offset += arrlen;

    INT_TA groups;UNPACK(buffer, offset, &groups, INT_TA_SIZE);offset += INT_TA_SIZE;
    int8_t isBias;UNPACK(buffer, offset, &isBias, INT8_T_SIZE);offset += UINT32_T_SIZE;
    uint32_t padding_mode;UNPACK(buffer, offset, &padding_mode, UINT32_T_SIZE);offset += UINT32_T_SIZE;


    // 检测 INT_TA nweights, INT_TA nbiases 
    layer_TA *l = netta._layers + idx;

    int8_t ret = make_conv_TA(l,
                              channel,
                              num,
                              size,
                              stride,
                              padding,
                              dilation,
                              groups,
                              isBias,
                              (PADDING_MODE_TA)padding_mode);
}
void make_norm_ta(void *tar, int32_t idx,
                  FLOATCA *mean, FLOATCA *variance,
                  uint32_t means, uint32_t variances) {

    unsigned char *buffer = (unsigned char *)tar;
    uint32_t offset = 0;

    INT_TA in_c;UNPACK(buffer, offset, &in_c, INT_TA_SIZE);offset += INT_TA_SIZE;
    FLOAT64_TA eps;UNPACK(buffer, offset, &eps, FLOAT64_TA_SIZE);offset += FLOAT64_TA_SIZE;
    FLOAT64_TA momentum;UNPACK(buffer, offset, &momentum, FLOAT64_TA_SIZE);offset += FLOAT64_TA_SIZE;
    int8_t affine;UNPACK(buffer, offset, &affine, INT8_T_SIZE);offset += UINT32_T_SIZE;
    int8_t track_running_stats;UNPACK(buffer, offset, &track_running_stats, INT8_T_SIZE);offset += UINT32_T_SIZE;

    layer_TA *l = netta._layers + idx;

    int8_t ret = make_norm_TA(l,
                              in_c,
                              eps,
                              momentum,
                              affine,
                              track_running_stats,
                              mean, variance,
                              means, variances);
}
void make_activ_ta(void *tar, int32_t idx) {
    unsigned char *buffer = (unsigned char *)tar;
    uint32_t offset = 0;
    uint32_t activ;UNPACK(buffer, offset, &activ, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    layer_TA *l = netta._layers + idx;

    int8_t ret = make_activ_TA(l, (ACTIVATION_TA)activ);
}

void make_tsops_ta(void *tar, int32_t idx) {
    unsigned char *buffer = (unsigned char *)tar;
    uint32_t offset = 0;
    uint32_t stn;UNPACK(buffer, offset, &stn, UINT32_T_SIZE);offset += UINT32_T_SIZE;
    INT_TA index2;UNPACK(buffer, offset, &index2, INT_TA_SIZE);offset += INT_TA_SIZE;
    int8_t kpIdx2;UNPACK(buffer, offset, &kpIdx2, INT8_T_SIZE);offset += UINT32_T_SIZE;
    INT_TA index1;UNPACK(buffer, offset, &index1, INT_TA_SIZE);offset += INT_TA_SIZE;
    int8_t kpIdx1;UNPACK(buffer, offset, &kpIdx1, INT8_T_SIZE);offset += UINT32_T_SIZE;

    layer_TA *l = netta._layers + idx;
    
    int8_t ret = make_tsops_TA(l, (STATION_TA)stn, index2, kpIdx2, index1, kpIdx1);
}

void make_pool_ta(void *tar, int32_t idx) {
    unsigned char *buffer = (unsigned char *)tar;
    uint32_t arrlen = INT_TA_SIZE * MAX_CONV_DIMENSIONS_TA;
    uint32_t offset = 0;

    INT_TA size[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, size, arrlen);offset += arrlen;
    INT_TA stride[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, stride, arrlen);offset += arrlen;
    INT_TA padding[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, padding, arrlen);offset += arrlen;
    INT_TA dilation[MAX_CONV_DIMENSIONS_TA];UNPACK(buffer, offset, dilation, arrlen);offset += arrlen;

    int8_t return_indices;UNPACK(buffer, offset, &return_indices, INT8_T_SIZE);offset += UINT32_T_SIZE;
    int8_t ceil_mode;UNPACK(buffer, offset, &ceil_mode, INT8_T_SIZE);offset += UINT32_T_SIZE;
    int8_t count_include_pad;UNPACK(buffer, offset, &count_include_pad, INT8_T_SIZE);offset += UINT32_T_SIZE;
    INT_TA divisor_override;UNPACK(buffer, offset, &divisor_override, INT_TA_SIZE);offset += INT_TA_SIZE;
    uint32_t padding_mode;UNPACK(buffer, offset, &padding_mode, UINT32_T_SIZE);offset += UINT32_T_SIZE;

    layer_TA *l = netta._layers + idx;

    int8_t ret = make_pool_TA(l,
                              size,
                              stride,
                              padding,
                              dilation,
                              return_indices,
                              ceil_mode,
                              count_include_pad,
                              divisor_override,
                              (PADDING_MODE_TA)padding_mode);
}
void make_linear_ta(void *tar, int32_t idx) {
    unsigned char *buffer = (unsigned char *)tar;
    uint32_t offset = 0;

    INT_TA in_c;UNPACK(buffer, offset, &in_c, INT_TA_SIZE);offset += INT_TA_SIZE;
    INT_TA out_c;UNPACK(buffer, offset, &out_c, INT_TA_SIZE);offset += INT_TA_SIZE;
    int8_t isBias;UNPACK(buffer, offset, &isBias, INT8_T_SIZE);offset += UINT32_T_SIZE;

    layer_TA *l = netta._layers + idx;

    int8_t ret = make_linear_TA(l,
                                in_c,
                                out_c,
                                isBias);
}

// =========================================================================
// =========================================================================

TEE_Result forward_network_ta(uint32_t param_types, TEE_Param params[4]) {

    // 这里传入的有 input 和将要执行的 layer 的 idx
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

	DMSG("forward_network_ta has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    // resetInOutPtr(&netta);
    memcpy(netta._input, params[0].memref.buffer, params[0].memref.size);
    
    // 这样处理相对来说比较简单 省去了后面的重置过程 
    // num : idx1 idx2 idx3 ...... 
    // 通过 num 确保访问不会越界 
    // params[1].memref.size / INT8_T_SIZE = MAX_LAYERS_SEQUENCE_TA
    memcpy(netta._index, params[1].memref.buffer, MAX_LAYERS_SEQUENCE_TA * INT32_T_SIZE);
    
    netta._tmp_outputs = params[0].memref.size;
    forward_network_TA(&netta);

    return TEE_SUCCESS;
}

TEE_Result forward_ret_network_ta(uint32_t param_types, TEE_Param params[4]) {

    // 这里传入的有 input 和将要执行的 layer 的 idx
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

	DMSG("forward_ret_network_ta has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    // 这里的 size 应该要等同于 Layer 中保存的真实 size
    memcpy(params[0].memref.buffer, netta._output, params[0].memref.size);
    return TEE_SUCCESS;
}

TEE_Result forwardFetch_network_ta(uint32_t param_types, TEE_Param params[4]) {

    // 这里传入的有 input 和将要执行的 layer 的 idx
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);

	DMSG("forwardFetch_network_ta has been called");

	if (param_types != exp_param_types)
		return TEE_ERROR_BAD_PARAMETERS;

    // resetInOutPtr(&netta);
    memcpy(netta._input, params[0].memref.buffer, params[0].memref.size);
    
    // 这样处理相对来说比较简单 省去了后面的重置过程 
    // num : idx1 idx2 idx3 ...... 
    // 通过 num 确保访问不会越界 
    // params[1].memref.size / INT8_T_SIZE = MAX_LAYERS_SEQUENCE_TA
    memcpy(netta._index, params[2].memref.buffer, MAX_LAYERS_SEQUENCE_TA * INT32_T_SIZE);
    
    netta._tmp_outputs = params[0].memref.size;
    forward_network_TA(&netta);
    // 这里的 size 应该要等同于 Layer 中保存的真实 size
    memcpy(params[1].memref.buffer, netta._output, params[1].memref.size);

    return TEE_SUCCESS;
}

