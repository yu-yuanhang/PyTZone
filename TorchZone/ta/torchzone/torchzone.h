#ifndef __TORCHZONE_H__
#define __TORCHZONE_H__

#include <network.h>
#include <kyee_test_ta.h>
#include <conv.h>
#include <activ.h>
#include <pool.h>
#include <norm.h>
#include <fullConn.h>
#include <tsops.h>

// extern network_TA netta;

// void printBytes(const void *ptr, size_t length);

// INT8_T_SIZE = 1
// sizeof(int32_t) = 4
// sizeof(size_t) = 8
// sizeof(float) = 4
// sizeof(ptr) = 8
// sizeof(int64_t) = 8


// 这里对于大量数据(并不包括 权重和偏移)的传递都是用连续的内存读取的方式
// 对于不同类型的数据顺序的存放在一块内存空间中 
// 但是对于内存的使用过程还没有做内存对齐的设计

TEE_Result make_network_ta_demo(uint32_t param_types, TEE_Param params[4]);
TEE_Result make_network_ta(uint32_t param_types, TEE_Param params[4]);
TEE_Result make_layer_ta(uint32_t param_types, TEE_Param params[4]);
TEE_Result make_layer_ext_ta(uint32_t param_types, TEE_Param params[4]);

void make_conv_ta(void *tar, int32_t idx);
void make_norm_ta(void *tar, int32_t idx,
                  FLOATCA *mean, FLOATCA *variance,
                  uint32_t means, uint32_t variances);
void make_activ_ta(void *tar, int32_t idx);
void make_pool_ta(void *tar, int32_t idx);
void make_linear_ta(void *tar, int32_t idx);
void make_tsops_ta(void *tar, int32_t idx);



TEE_Result forward_network_ta(uint32_t param_types, TEE_Param params[4]);
TEE_Result forward_ret_network_ta(uint32_t param_types, TEE_Param params[4]);
TEE_Result forwardFetch_network_ta(uint32_t param_types, TEE_Param params[4]);


#endif