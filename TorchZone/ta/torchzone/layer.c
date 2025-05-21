#include "layer.h"
#include "network.h"
#include <conv.h>
#include <activ.h>
#include <norm.h>
#include <fullConn.h>
#include <pool.h>
#include <tsops.h>

#if CHECK
void printLayer(layer_TA *layerta) {
    printf("!!!!!!!!!!!!!!!!! Layer idx = %d !!!!!!!!!!!!!!!!!\n", layerta->_idx); 
    printf("type == %u : dimsNum == %u\n", 
           layerta->_type,
           layerta->_dimsNum);
    printf("workspace_size == %d : batch == %d\n", 
           layerta->_workspace_size,
           layerta->_batch);
    printf("binary == %d : xnor == %d\n", 
           layerta->_binary,
           layerta->_xnor);
    printf("inputs == %d : outputs == %d\n", 
           layerta->_inputs,
           layerta->_outputs);
    printf("nweights == %d\nnbiases == %d\n", 
           layerta->_nweights,
           layerta->_nbiases);
    
    printf("dims     : ");
    for (INT_TA i = 0; i <= layerta->_dims[0]; ++i) printf("%u ", layerta->_dims[i]);
    printf("\n");
    printf("out_dims : ");
    for (INT_TA i = 0; i <= layerta->_out_dims[0]; ++i) printf("%u ", layerta->_out_dims[i]);
    printf("\n");
    if (NULL != layerta->_weights) {       
        printf("============================\nweights ...\n");
        for (INT_TA i = 0; i < layerta->_nweights && i < 40; ++i) {
            if (!(i % 10) && 0 != i) printf("\n");
            // printf("%+.4f  ", layerta->_weights[i]); // "+":始终显示符号
            printf("%u  ", (INT_TA)(layerta->_weights[i]));
        }
        printf("\n");
    }
    if (NULL != layerta->_biases) {
        printf("============================\nbiases ...\n");
        for (INT_TA i = 0; i < layerta->_nbiases && i < 40; ++i) {
            if (!(i % 10) && 0 != i) printf("\n");
            // printf("%+.4f  ", layerta->_biases[i]);
            printf("%u  ", (INT_TA)(layerta->_biases[i]));
        }
        printf("\n");
    }
    if (BATCHNORM_TYPE == layerta->_type) {
        if (NULL != layerta->_mean) {       
            printf("============================\nmean ...\n");
            for (INT_TA i = 0; i < layerta->_nweights && i < 40; ++i) {
                if (!(i % 10) && 0 != i) printf("\n");
                // printf("%+.4f  ", layerta->_weights[i]); // "+":始终显示符号
                printf("%u  ", (INT_TA)(layerta->_mean[i]));
            }
            printf("\n");
        }
        if (NULL != layerta->_variance) {
            printf("============================\nvariance ...\n");
            for (INT_TA i = 0; i < layerta->_nweights && i < 40; ++i) {
                if (!(i % 10) && 0 != i) printf("\n");
                // printf("%+.4f  ", layerta->_biases[i]);
                printf("%u  ", (INT_TA)(layerta->_variance[i]));
            }
            printf("\n");
        }
    }
    switch (layerta->_type) {
        case CONV_TYPE:
            printConv(layerta);
            break;
        case MAXPOOL_TYPE:
            printPool(layerta);
            break;
        case FCONNECTED_TYPE:
            printLinear(layerta);
            break;
        case BATCHNORM_TYPE:
            printNorm(layerta);
            break;
        case ACTIV_TYPE:
            printActiv(layerta);
            break;
        case TSTATION_TYPE:
            printTsops(layerta);
            break;
        // ......
        
        default: 
	        break;

    }
}

INT_TA getKernelSize(const layer_TA * const layerta) {
    uint32_t size = 1;
    for (uint32_t i = 0; i < layerta->_dimsNum - 1; ++i)
        size *= layerta->_size[i];
    return size;
}
INT_TA getOutSize_c(const layer_TA * const layerta) {
    INT_TA size = 1;
    for (INT_TA i = 2; i < layerta->_out_dims[0] + 1; ++i) 
        size *= layerta->_out_dims[i];
    return size;
}
INT_TA getInSize_c(const layer_TA * const layerta) {
    INT_TA size = 1;
    for (INT_TA i = 2; i < layerta->_dims[0] + 1; ++i) 
        size *= layerta->_dims[i];
    return size;
}

// 这个函数使用时候还是会有风险
// 对于 &output == &input 的情况需要将 keep 操作分开 
int dataKeep(layer_TA *l, struct network_TA *net) {
    INT_TA batch = l->_batch;
    if (l->_keepIn) {
        FLOATTA *input = l->_input;
        INT_TA inputs_size = l->_inputs * batch;
        if (NULL != input && 0 != inputs_size) {
        //     input = (FLOATTA *)aligned_malloc(IntTUint32(inputs_size) * FLOAT_TA_SIZE, ALIGNMENT);
        //     if (NULL == input) {
        //         EMSG("malloc for input error : inputs_size = %d\n", inputs_size);
        //         return -1;
        //     }
            memcpy(input, net->_input, inputs_size * FLOAT_TA_SIZE);
        }
    }
    if (l->_keepOut) {
        FLOATTA *output = l->_output;
        INT_TA outputs_size = l->_outputs * batch;
        if (NULL != output && 0 != outputs_size) {
            // output = (FLOATTA *)aligned_malloc(IntTUint32(outputs_size) * FLOAT_TA_SIZE, ALIGNMENT);
            // if (NULL == output) {
            //     EMSG("malloc for output error : outputs_size = %d\n", outputs_size);
            //     return -1;
            // }
            memcpy(output, net->_output, outputs_size * FLOAT_TA_SIZE);
        }
    }

    return 0;
}


#endif
