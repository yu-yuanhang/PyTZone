#include "norm.h"

int8_t make_norm_TA(layer_TA *layerta,
                    INT_TA in_c,
                    FLOAT64_TA eps,
                    FLOAT64_TA momentum,
                    int8_t affine,
                    int8_t track_running_stats,
                    FLOATCA *mean, FLOATCA *variance,
                    uint32_t means, uint32_t variances) {
    DMSG("make_norm_TA has been called");

    // layerta->_forward_ta = forward_conv2d_TA;
    layerta->_forward_ta = forward_batchnorm2d_TA;

    // layerta->in_c = in_c;
    ERROR_CHECK_RET_TA (layerta->_dims[1] == in_c, false, "make_norm_TA : in_c error");
    layerta->_eps = eps;
    layerta->_momentum = momentum;
    layerta->_affine = affine;
    layerta->_track_running_stats = track_running_stats;

    if(BATCHNORM_TYPE == layerta->_type) {
        if (NULL != mean && 0 != means) {
            layerta->_mean = (FLOATTA *)aligned_malloc(IntTUint32(means) * FLOAT_TA_SIZE, ALIGNMENT);
            if (NULL == layerta->_mean) {
                EMSG("malloc for mean error : means = %u\n", means);
                return -1;
            }
            memcpy(layerta->_mean, mean, means * FLOAT_TA_SIZE);
        }
        if (NULL != variance && 0 != variances) {
            layerta->_variance = (FLOATTA *)aligned_malloc(IntTUint32(variances) * FLOAT_TA_SIZE, ALIGNMENT);
            if (NULL == layerta->_variance) {
                EMSG("malloc for variance error : variances = %u\n", variances);
                return -1;
            } 
            // memcpy(layerta->_variance, variance, variances * FLOAT_TA_SIZE);
            // ==============================
            for (uint32_t i = 0; i < variances; ++i) {layerta->_variance[i] = (ta_sqrt(variance[i]) + eps);}
            // ==============================
        }
    }
    
    return 0;    
}
void forward_batchnorm2d_TA(base_layer_TA *l, network_TA *net)
{
    printf("forward_batchnorm2d_TA(base_layer_TA *, network_TA *)\n");
    // if(l->type == BATCHNORM_TYPE) copy_cpu(l->outputs*l->batch, net.input, 1, l->output, 1);
    // copy_cpu(l->outputs*l->batch, l->output, 1, l->x, 1);
    int wh = getOutSize_c(l);
    if(l->_type == BATCHNORM_TYPE) swapInOutPtr(net);
    if(net->_train){
        // mean_cpu(l->output, l->batch, l->out_c, l->out_h*l->out_w, l->mean);
        // variance_cpu(l->output, l->mean, l->batch, l->out_c, l->out_h*l->out_w, l->variance);

        // scal_cpu(l->out_c, .99, l->rolling_mean, 1);
        // axpy_cpu(l->out_c, .01, l->mean, 1, l->rolling_mean, 1);
        // scal_cpu(l->out_c, .99, l->rolling_variance, 1);
        // axpy_cpu(l->out_c, .01, l->variance, 1, l->rolling_variance, 1);

        // normalize_cpu(l->output, l->mean, l->variance, l->batch, l->out_c, l->out_h*l->out_w);
        // copy_cpu(l->outputs*l->batch, l->output, 1, l->x_norm, 1);
    } else {
        normalize_cpu_TA(net->_output, l->_mean, l->_variance, l->_batch, l->_out_dims[1], wh);
    }
    scale_bias_TA(net->_output, l->_weights, l->_batch, l->_out_dims[1], wh);
    add_bias_TA(net->_output, l->_biases, l->_batch, l->_out_dims[1], wh);
}
void normalize_cpu_TA(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(variance[f]);
                // x[index] = (x[index] - mean[f])/(ta_sqrt(variance[f]) + .000001f);
            }
        }
    }
}

#if CHECK           
void printNorm(const layer_TA * const layerta) {
    printf("~~~~~~~~~~~~~ for norm ~~~~~~~~~~~~~\n");
    printf("in_c == %d\n", layerta->_dims[1]);
    printf("eps == %f : momentum == %f\n", 
           layerta->_eps,
           layerta->_momentum);
    printf("affine == %d\ntrack_running_stats == %d\n", 
           layerta->_affine,
           layerta->_track_running_stats);
    printf("dimsNum == %u\n", layerta->_dimsNum);
}
#endif