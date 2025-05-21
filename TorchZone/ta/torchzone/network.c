#include "network.h"

network_TA netta;

int8_t make_network_TA(INT_TA num, INT_TA batch, 
                       INT_TA inputs, INT_TA outputs,
                       INT_TA inoutSize,
                       INT_TA workspace_size,
                       FLOAT64_TA clip,
                       int8_t train) {
    // 目前还是考虑将 makeNet 和 makeLayer 分开

    netta._num = num;
    netta._batch = batch;
    netta._inputs = inputs;
    netta._outputs = outputs;
    netta._inoutSize = inoutSize;
    netta._workspace_size = workspace_size;

    netta._input = NULL;
    netta._output = NULL;

    // 判断逻辑交给 CA 侧
    // netta._inoutSize = (inputs > outputs) ? inputs : outputs;
    netta._mem1 = aligned_malloc(IntTUint32(inoutSize * batch) * FLOAT_TA_SIZE, ALIGNMENT);
    netta._input = netta._mem1;
    netta._mem2 = aligned_malloc(IntTUint32(inoutSize * batch) * FLOAT_TA_SIZE, ALIGNMENT);
    netta._output = netta._mem2;

    // netta._workspace = params[i].memref.buffer;
    // netta._workspace = NULL;
    netta._workspace = (FLOATTA *)aligned_malloc(IntTUint32(workspace_size), ALIGNMENT);
    if (NULL == netta._workspace) {
        EMSG("malloc for workspace error : workspace_size = %u\n", workspace_size);
        return -1;
    }

    netta._clip = clip;
    netta._train = train;
    
    // tobe free
    // netta._layers = calloc(num, sizeof(layer_TA));
    memset(netta._layers, 0, sizeof(netta._layers));

    // index 指向带执行的 layer 目前在初始化阶段是空的
    // 这里也可以直接 memset 赋空值 
    // 这里处于安全考虑就直接 memset 代替
    memset(netta._index, 0, sizeof(netta._index));
    // netta._idx = -1;
    
    // ============================================================
    netta._tmp_outputs = 0;
    return 0;
}

#if CHECK
void printNet(network_TA *pNetTA) {
    printf("============== network_TA net ============== \nnum == %u : batch == %u\n", 
           pNetTA->_num,
           pNetTA->_batch);
    printf("inputs == %u : outputs == %u\n", 
           pNetTA->_inputs,
           pNetTA->_outputs);
    printf("nworkspace_size == %u\ninoutSize = %u\n", 
           pNetTA->_workspace_size,
           pNetTA->_inoutSize);
    
    // ......
}
#endif


void forward_network_TA(network_TA *pNetTA) {
    // if (pNetTA->_workspace_size) {
    //     DMSG("workspace_size=%d", pNetTA->_workspace_size);
    //     pNetTA->_workspace = malloc((pNetTA->_workspace_size) * FLOAT_TA_SIZE);
    //         if (NULL == pNetTA->_workspace) {
    //             EMSG("malloc for workspace error : workspace_size = %d\n", pNetTA->_workspace_size);
    //             return;
    //         }
    //     // memset(pNetTA->_workspace, 0, pNetTA->_workspace_size);
    // }
    for (int32_t i = 1; i <= pNetTA->_index[0]; ++i) {
        layer_TA l = pNetTA->_layers[pNetTA->_index[i]];
        DMSG("l.forward_ta(...) : idx = %d", l._idx);
        l._forward_ta(&l, pNetTA);
        swapInOutPtr(pNetTA);
        if (TSTATION_TYPE <= l._type) continue;
        pNetTA->_tmp_outputs = l._outputs;
    }
    swapInOutPtr(pNetTA);

}

void swapInOutPtr(network_TA *pNetTA) {
    FLOATTA *tmp = pNetTA->_input;
    pNetTA->_input = pNetTA->_output;
    pNetTA->_output = tmp;
}
void resetInOutPtr(network_TA *pNetTA) {
    pNetTA->_input = pNetTA->_mem1;
    pNetTA->_output = pNetTA->_mem2;
}