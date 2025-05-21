#include "pool.h"


int8_t make_pool_TA(layer_TA *layerta,
                    INT_TA size[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA stride[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA padding[/* MAX_CONV_DIMENSIONS_TA */],
                    INT_TA dilation[/* MAX_CONV_DIMENSIONS_TA */],
                    int8_t return_indices,
                    int8_t ceil_mode,
                    int8_t count_include_pad,
                    INT_TA divisor_override,
                    PADDING_MODE_TA padding_mode) {
    DMSG("make_pool_TA has been called");
    // ... 取决于 l->_dimsNum
    // layerta->_forward_ta = forward_conv2d_TA;
    if (MAXPOOL_TYPE == layerta->_type)
        layerta->_forward_ta = forward_maxpool2d_TA;
    else if (MAXPOOL_TYPE == layerta->_type) 
        layerta->_forward_ta = forward_avgpool2d_TA;

    memcpy(layerta->_size, size, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    memcpy(layerta->_stride, stride, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    memcpy(layerta->_padding, padding, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    memcpy(layerta->_dilation, dilation, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);

    layerta->_return_indices = return_indices;
    layerta->_ceil_mode = ceil_mode;
    layerta->_count_include_pad = count_include_pad;
    layerta->_divisor_override = divisor_override;
    layerta->_padding_mode = padding_mode;

    return 0;
}

void forward_maxpool2d_TA(base_layer_TA *l, network_TA *net) {
    printf("forward_maxpool2d_TA(base_layer_TA *, network_TA *)\n");
    int b,i,j,k,m,n;
    int h_offset = -l->_padding[0];
    int w_offset = -l->_padding[1];

    int h_out = l->_out_dims[2];
    int w_out = l->_out_dims[3];
    int c = l->_dims[1];

    for(b = 0; b < l->_batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h_out; ++i){
                for(j = 0; j < w_out; ++j){
                    int out_index = j + w_out*(i + h_out*(k + c*b));
                    float max = -first_aim_money;
                    // int max_i = -1;
                    for(n = 0; n < l->_size[0]; ++n){
                        for(m = 0; m < l->_size[1]; ++m){
                            int cur_h = h_offset + i*l->_stride[0] + n;
                            int cur_w = w_offset + j*l->_stride[1] + m;
                            int index = cur_w + l->_dims[3]*(cur_h + l->_dims[2]*(k + b*l->_dims[1]));
                            int valid = (cur_h >= 0 && cur_h < l->_dims[2] &&
                                         cur_w >= 0 && cur_w < l->_dims[3]);
                            float val = (valid != 0) ? net->_input[index] : -first_aim_money;
                            // max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    net->_output[out_index] = max;
                    // l->indexes[out_index] = max_i;
                }
            }
        }
    }
}

void forward_avgpool2d_TA(base_layer_TA *l, network_TA *net) {
    int b,i,k;
    int c = l->_dims[1];

    for(b = 0; b < l->_batch; ++b){
        for(k = 0; k < c; ++k){
            int out_index = k + b*c;
            net->_output[out_index] = 0;
            for(i = 0; i < l->_dims[2]*l->_dims[3]; ++i){
                int in_index = i + l->_dims[2]*l->_dims[3]*(k + b*c);
                net->_output[out_index] += net->_input[in_index];
            }
            net->_output[out_index] /= l->_dims[2]*l->_dims[3];
        }
    }
}

#if CHECK           
void printPool(const layer_TA * const layerta) {
    printf("~~~~~~~~~~~~~ for pool ~~~~~~~~~~~~~\n");
    printf("return_indices == %d : ceil_mode == %d\n", 
           layerta->_return_indices,
           layerta->_ceil_mode);
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