#include "net.h"

void make_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                     INT_TA num, INT_TA batch,
                     INT_TA inputs, INT_TA outputs,
                     INT_TA inoutSize,
                     INT_TA workspace_size,
                     FLOAT64_TA clip,
                     int8_t train) {

    // 理论上 make_network_ca 是需要被调用的地一个接口

    printf("make_network_ca(...)\n");

    // 这里 share Mem 的安全性需要进一步封装
    // if (!sm.buffer || sm.size < MAKE_NETWORK_LENGTH) {
    //     // EMSG("malloc for buffer error : size = %d", MAKE_NETWORK_LENGTH);
    //     // printf("share memory allocation error : make_network_ca(...) : sm.size = %d\n", sm.size);
    //     errx(1, "share memory allocation error : make_network_ca(...) : sm.size = %d",
    //          sm.size);
    // }
    
    void *buffer = (void *)malloc(MAKE_NETWORK_LENGTH);

    uint32_t offset = 0;
    PACK(buffer, offset, &num, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &batch, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &inputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &outputs, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &inoutSize, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &workspace_size, INT_TA_SIZE);offset += INT_TA_SIZE;
    PACK(buffer, offset, &clip, FLOAT64_TA_SIZE);offset += FLOAT64_TA_SIZE;
    PACK(buffer, offset, &train, INT8_T_SIZE);offset += UINT32_T_SIZE;

    TEEC_Operation op;
    memset(&op, 0, sizeof(op));
    uint32_t err_origin;
    TEEC_Result res;

    // 这里目前不再使用 shareMem 统一用 临时内存代替
    // 主要原因是 shareMem 不适用与多线程调用的场景
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, 
                                     TEEC_NONE,
                                     TEEC_NONE, 
                                     TEEC_NONE);

    op.params[0].tmpref.buffer = buffer;
	op.params[0].tmpref.size = MAKE_NETWORK_LENGTH;

    // op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, 
    //                                  TEEC_NONE,
    //                                  TEEC_NONE, 
    //                                  TEEC_NONE);
    // op.params[0].memref.parent = &sm;
    // op.params[0].memref.offset = 0;
    // op.params[0].memref.size = MAKE_NETWORK_LENGTH;

    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), MAKE_NETWORK_CMD, &op, &err_origin);

    free(buffer);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_NET) failed 0x%x origin 0x%x",
         res, err_origin);
}

int8_t forward_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                          FLOATCA *input, INT_TA length, 
                          int32_t idx[/* MAX_LAYERS_SEQUENCE_TA */]) {
    printf("forward_network_ca(...)\n");
    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    // if (length * FLOAT_TA_SIZE > sm.size) {
    //     // EMSG("forward_network_ca : length > size of shareMem");
    //     printf("forward_network_ca : length > size of shareMem\n");
    //     return -1;
    // }
    // resetSM();
    // memcpy(sm.buffer, input, length * FLOAT_TA_SIZE);

	memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, 
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, 
                                     TEEC_NONE);
    
    op.params[0].tmpref.buffer = input;
	op.params[0].tmpref.size = length * FLOAT_TA_SIZE;

    op.params[1].tmpref.buffer = idx;
    op.params[1].tmpref.size = INT32_T_SIZE * MAX_LAYERS_SEQUENCE_TA; 

    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), FORWARD_CMD, &op, &err_origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FORWARD_NET) failed 0x%x origin 0x%x",
         res, err_origin);

    return 0;
}

int8_t forward_ret_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                              FLOATCA *output, INT_TA length) {
    printf("forward_ret_network_ca(...)\n");
    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    // if (length * FLOAT_TA_SIZE > sm.size) {
    //     // EMSG("forward_ret_network_ca : length > size of shareMem");
    //     printf("forward_ret_network_ca : length > size of shareMem\n");
    //     return -1;
    // }

	op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE,
					                 TEEC_NONE, 
                                     TEEC_NONE);
    op.params[0].tmpref.buffer = output;
	op.params[0].tmpref.size = length * FLOAT_TA_SIZE;

    printf("length = %d\n", length);

    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), FORWARD_RET_CMD, &op, &err_origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FORWARD_RET_NET) failed 0x%x origin 0x%x",
         res, err_origin);

    return 0;
}

int8_t forwardFetch_network_ca(TEEC_INVITATION_T *TEEC_INVITATION,
                               FLOATCA *input, INT_TA length_in, 
                               FLOATCA *output, INT_TA length_out, 
                               int32_t idx[/* MAX_LAYERS_SEQUENCE_TA */]) {

    printf("forwardFetch_network_ca(...)\n");
    TEEC_Operation op;
    uint32_t err_origin;
    TEEC_Result res;

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, 
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_MEMREF_TEMP_INPUT, 
                                     TEEC_NONE);

    op.params[0].tmpref.buffer = input;
	op.params[0].tmpref.size = length_in * FLOAT_TA_SIZE;

    op.params[1].tmpref.buffer = output;
	op.params[1].tmpref.size = length_out * FLOAT_TA_SIZE;

    op.params[2].tmpref.buffer = idx;
    op.params[2].tmpref.size = INT32_T_SIZE * MAX_LAYERS_SEQUENCE_TA; 

    res = TEEC_InvokeCommand(&(TEEC_INVITATION->sess), FORWARD_FETCH_CMD, &op, &err_origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FORWARD_FETCH_CMD) failed 0x%x origin 0x%x",
         res, err_origin);

    return 0;

}