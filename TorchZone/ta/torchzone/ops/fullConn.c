#include "fullConn.h"

int8_t make_linear_TA(layer_TA *layerta,
                      INT_TA in_c,
                      INT_TA out_c,
                      int8_t isBias) {
    DMSG("make_linear_TA has been called");

    layerta->_forward_ta = forward_linear_TA;

    // layerta->in_c = in_c;
    ERROR_CHECK_RET_TA (layerta->_dims[1] == in_c, false, "make_linear_TA : in_c error");
    ERROR_CHECK_RET_TA (layerta->_out_dims[1] == out_c, false, "make_linear_TA : out_c error");
    layerta->_isBias = isBias;
    
    return 0;
}


void forward_linear_TA(base_layer_TA *l, network_TA *net) {
    
    printf("forward_linear_TA(base_layer_TA *, network_TA *)\n");

    // fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    // int m = l.batch;
    // int k = l.inputs;
    // int n = l.outputs;
    // float *a = net.input;
    // float *b = l.weights;
    // float *c = l.output;

    int m = l->_batch;
    int k = l->_inputs;
    int n = l->_outputs;
    float *a = net->_input;
    float *b = l->_weights;
    float *c = net->_output;
    
    gemm_TA(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if (l->_isBias) add_bias_TA(net->_output, l->_biases, l->_batch, l->_outputs, 1);
    dataKeep(l, net);
}

#if CHECK           
void printLinear(const layer_TA * const layerta) {
    printf("~~~~~~~~~~~~~ for linear ~~~~~~~~~~~~~\n");
    printf("isBias == %d\n", layerta->_isBias);
    printf("in_c == %d\nout_c == %d\n", 
           layerta->_dims[1],
           layerta->_out_dims[1]);
}
#endif
