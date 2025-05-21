#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <layer.h>
#include <stdlib.h>

typedef struct network_TA { 
    
    INT_TA _num;
    INT_TA _batch;
    
    INT_TA _inputs, _outputs;   // 输入数量和输出数量的最大值
    // INT_TA _high, _weight, _channel;
    // uint32_t 最高 4GB 但是在 TEE 目前应该够用 
    // sizeof(int32_t) = 4
    // sizeof(size_t) = 8
    // sizeof(float) = 4
    // sizeof(ptr) = 8
    // sizeof(int) = 4
    INT_TA _inoutSize;
    INT_TA _workspace_size;  

    FLOATTA *_input;              // 输入数据 或作为中间变量   
    FLOATTA *_output;
    FLOATTA *_mem1;
    FLOATTA *_mem2;    

    FLOATTA *_workspace;          // 网络的工作空间 用于存储临时变量以进行计算

    // 剪裁: 是用来限制网络输出范围的参数 通常用于防止输出值过大或过小
    FLOAT64_TA _clip;    
    int8_t _train;

    // Layers 
    // layer_TA *_layers;
    layer_TA _layers[MAX_LAYERS_SEQUENCE_TA];
    // num : idx1 idx2 idx3 ......
    int32_t _index[MAX_LAYERS_SEQUENCE_TA];
    // 指向当前已经被执行完成的最后一个 layer 的下标 
    // int8_t _idx;

    // =========================================================
    INT_TA _tmp_outputs;

} network_TA;

int8_t make_network_TA(INT_TA num, INT_TA batch, 
                       INT_TA inputs, INT_TA outputs,
                       INT_TA inoutSize,
                       INT_TA workspace_size,
                       FLOAT64_TA clip,
                       int8_t train);
#if CHECK
void printNet(network_TA *pNetTA);
#endif

void forward_network_TA(network_TA *pNetTA);
void swapInOutPtr(network_TA *pNetTA);
void resetInOutPtr(network_TA *pNetTA);


#endif