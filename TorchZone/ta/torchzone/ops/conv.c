#include "conv.h"

int8_t make_conv_TA(layer_TA *layerta,
                    INT_TA channel,
                    INT_TA num,
                    INT_TA size[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA stride[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA padding[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA dilation[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA groups,
                    int8_t isBias,
                    PADDING_MODE_TA padding_mode) {

    DMSG("make_conv_TA has been called");
    // layerta->_forward_ta = ... 取决于 l->_dimsNum
    layerta->_forward_ta = forward_conv2d_TA;

    layerta->_channel = channel;
    layerta->_num = num;
    
    // 为了传惨的逻辑尽量简单
    // TA 内部的成员结构信息都尽量相同 这里参数的维度信息通过 DIMENSIONALITY_TA 体现
    // layerta->_size = malloc((dimsNum - 1) * INT_TA_SIZE);
    memcpy(layerta->_size, size, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    // layerta->_stride = malloc((dimsNum - 1) * INT_TA_SIZE);
    memcpy(layerta->_stride, stride, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    // layerta->_padding = malloc((dimsNum - 1) * INT_TA_SIZE);
    memcpy(layerta->_padding, padding, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    // layerta->_dilation = malloc((dimsNum - 1) * INT_TA_SIZE);
    memcpy(layerta->_dilation, dilation, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);

    layerta->_groups = groups;
    layerta->_isBias = isBias;
    layerta->_padding_mode = padding_mode;
    
    return 0;
}

#if CHECK           
void printConv(const layer_TA * const layerta) {
    printf("~~~~~~~~~~~~~ for conv ~~~~~~~~~~~~~\n");
    printf("channel == %d : num == %d\n", 
           layerta->_channel,
           layerta->_num);
    printf("groups == %d\nisBias == %d\n", 
           layerta->_groups,
           layerta->_isBias);
    printf("padding_mode == %u\ndimsNum == %u\n", 
           layerta->_padding_mode,
           layerta->_dimsNum);

    printf("size     : ");
    for (uint32_t i = 0; i < layerta->_dimsNum - 1; ++i) printf("%u ", layerta->_size[i]);
    printf("\n");
    printf("stride   : ");
    for (uint32_t i = 0; i < layerta->_dimsNum - 1; ++i) printf("%u ", layerta->_stride[i]);
    printf("\n");
    printf("padding  : ");
    for (uint32_t i = 0; i < layerta->_dimsNum - 1; ++i) printf("%u ", layerta->_padding[i]);
    printf("\n");
    printf("dilation : ");
    for (uint32_t i = 0; i < layerta->_dimsNum - 1; ++i) printf("%u ", layerta->_dilation[i]);
    printf("\n");
}
#endif

void forward_conv2d_TA(base_layer_TA *l, network_TA *net) {

    // fill_cpu_TA(l->_outputs * l->_batch, 0, l->output, 1);
    // fill_cpu_zero_TA(l->_output, l->_outputs * l->_batch);

    // fill_cpu_zero_TA(net->_output, l->_outputs * l->_batch);

    /*
    if(l.xnor){
       // ......
    }
    */    
    printf("forward_conv2d_TA(base_layer_TA *, network_TA *)\n");
    int channels = l->_channel / l->_groups;
    int m = l->_num / l->_groups;
    int k = getKernelSize(l) * channels;
    int n = getOutSize_c(l);
    for(int i = 0; i < l->_batch; ++i){
        for(int j = 0; j < l->_groups; ++j){
            // weight *
            FLOATTA *a = l->_weights + (j * l->_nweights / l->_groups);
            // workplace *
            FLOATTA *data_col = net->_workspace;
            // 当前分组的输出
            // FLOATTA *c = l->_output + (i * uint32ToInt(l->_groups) + j) * n * m;
            FLOATTA *c = net->_output + (i * l->_groups + j) * n * m;
            // 当前的输入特征图
            FLOATTA *data_im =  net->_input + (i * l->_groups + j) * channels * getInSize_c(l);
            if (l->_size[0] == 1 && l->_size[1] == 1) {
                data_col = data_im;
            } else {
                // 将输入的特征图展开为列形式
                // 这里进行 pad 操作理论上需要进行 padding_mode 的逻辑判断
                // 目前就简单的进行补 0
                // void im2col_cpu_TA(FLOATTA* data_im,
                //                 int channels,  int height,  int width,
                //                 int ksize,  int stride, int pad, FLOATTA* data_col);
                // im2col_cpu_TA(data_im, 
                //               channels, l->_dims[2], l->_dims[3],
                //               l->_size[0],
                //               l->_stride[0],
                //               l->_padding[0],
                //               data_col);
                im2col_cpu_TA_2d(data_im, 
                              channels, l->_dims[2], l->_dims[3],
                              l->_size[0], l->_size[1],
                              l->_stride[0], l->_stride[1],
                              l->_padding[0], l->_padding[1],
                              data_col);
            }
            // ...... todo dimsNum
            // 输出保存在 FLOATTA *c 中
            gemm_TA(0,0,m,n,k,1,a,k,data_col,n,1,c,n);
        }
    }

    // if(l->batch_normalize){
    //     forward_batchnorm_layer_TA(l, net);
    // } else {
    //     add_bias_TA(l->output, l->biases, l->batch, l->n, l->out_h*l->out_w);
    // }
    if (l->_isBias) add_bias_TA(net->_output, l->_biases, l->_batch, l->_num, n);
    
    // activate_array_TA(l->output, l->outputs*l->batch, l->activation);
    // if(l->binary || l->xnor) swap_binary_TA(&l);
    dataKeep(l, net);
}