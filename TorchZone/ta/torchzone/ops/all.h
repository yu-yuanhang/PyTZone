#ifndef __ALL_H__
#define __ALL_H__

#include <network.h>
#include <im2col_TA.h>
#include <gemm_TA.h>
// #include <fdlibm.h>

#include <stdint.h>

extern network_TA netta;

// 以下对于 Layer 的 make 过程这里就简单的逐层进行 make

static inline int8_t initLayer(layer_TA *layerta, 
                             // network_TA *netta,
                             LAYER_TYPE_TA type, 
                             DIMENSIONALITY_TA dimsNum,
                             INT_TA dims[/* MAX_CONV_DIMENSIONS_TA */],
                             INT_TA out_dims[/* MAX_CONV_DIMENSIONS_TA */],
                             int8_t binary, int8_t xnor,
                             int8_t keepIn, int8_t keepOut,
                             INT_TA batch,
                             INT_TA inputs, INT_TA outputs,
                            //  FLOATTA *weights, FLOATTA *biases,
                             INT_TA nweights, INT_TA nbiases,
                             INT_TA workspace_size,
                             int32_t idx,
                             FLOATTA *weights, FLOATTA *biases) {

    memset(layerta, 0, sizeof(layer_TA));
    layerta->net = &netta;
    layerta->_forward_ta = NULL;
    layerta->_activate_TA = NULL;

    // layerta->net = netta;
    layerta->_type = type;
    layerta->_dimsNum = dimsNum;

    // 考虑到后续的可扩展性
    // 这里尽量提高代码的可重用行 即使是 TA 中也保存了维度 num
    // layerta->_dims = malloc((dims[0] + 1) * INT_TA_SIZE);
    memcpy(layerta->_dims, dims, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);
    // layerta->_out_dims = malloc((out_dims[0] + 1) * INT_TA_SIZE);
    memcpy(layerta->_out_dims, out_dims, MAX_CONV_DIMENSIONS_TA * INT_TA_SIZE);

    layerta->_binary = binary;
    layerta->_xnor = xnor;

    layerta->_keepIn = keepIn;
    layerta->_keepIn = keepOut;

    layerta->_batch = batch;
    layerta->_inputs = inputs;
    layerta->_outputs = outputs;

    layerta->_nweights = nweights;
    layerta->_nbiases = nbiases;
    layerta->_workspace_size = workspace_size;
    layerta->_idx = idx;

    // 以下指针默认初始化
    layerta->_mean = NULL;
    layerta->_variance = NULL;
    // layerta->_weights = calloc(nweights, FLOAT_TA_SIZE);
    // layerta->_biases = calloc(nbiases, FLOAT_TA_SIZE);
    layerta->_weights = NULL;
    layerta->_biases = NULL;

    // 对于没一层的中间结果其实可以选择不保留
    // 具体的优化策略还需要进一步设计
    // 这里实验阶段暂时在暂时不保留所有的中间结果 所以这个字段暂时是闲置的
    layerta->_input = NULL;
    layerta->_output = NULL;
    // layerta->_output = calloc(outputs, FLOAT_TA_SIZE);
    layerta->_indexes = NULL;

// ===========================================================
    if (NULL != weights && 0 != nweights) {
        layerta->_weights = (FLOATTA *)aligned_malloc(IntTUint32(nweights) * FLOAT_TA_SIZE, ALIGNMENT);
        if (NULL == layerta->_weights) {
            EMSG("malloc for weights error : nweights = %u\n", layerta->_nweights);
            return -1;
        }
        memcpy(layerta->_weights, weights, nweights * FLOAT_TA_SIZE);
    }
    if (NULL != biases && 0 != nbiases) {
        layerta->_biases = (FLOATTA *)aligned_malloc(IntTUint32(nbiases) * FLOAT_TA_SIZE, ALIGNMENT);
        if (NULL == layerta->_biases) {
            EMSG("malloc for biases error : nbiases = %u\n", layerta->_nbiases);
            return -1;
        } 
        memcpy(layerta->_biases, biases, nbiases * FLOAT_TA_SIZE);
    }
    if (keepIn) {
        INT_TA inputs_size = inputs * batch;
        layerta->_input = (FLOATTA *)aligned_malloc(IntTUint32(inputs_size) * FLOAT_TA_SIZE, ALIGNMENT);
        if (NULL == layerta->_input) {
            EMSG("malloc for input error : inputs_size = %d\n", inputs_size);
            return -1;
        }
    }
    if (keepOut) {
        INT_TA outputs_size = outputs * batch;
        layerta->_output = (FLOATTA *)aligned_malloc(IntTUint32(outputs_size) * FLOAT_TA_SIZE, ALIGNMENT);
        if (NULL == layerta->_output) {
            EMSG("malloc for output error : outputs_size = %d\n", outputs_size);
            return -1;
        }
    }
// ===========================================================
}

#define base_layer_TA layer_TA
// ===========================================================
/* math.c */
#define PI 3.14159265358979323846
#define e  2.7182818284590452354
#define ln_2 0.69314718055994530942
#define ln_10 2.30258509299404568402
#define first_aim_money 1000000000.0f

#define fabs(a) ((a)>0?(a):(-(a)))

float ta_max(float a, float b);
double ta_pow(double a,int n);
double ta_eee(double x);
double ta_exp(double x);
float ta_rand();
int ta_floor(double x);
double ta_sqrt(double x);
double ta_ln(double x);
double ta_log(double a,double N);
double ta_sin(double x);
double ta_cos(double x);
double ta_tan(double x);

void reverse(char *str, int len);
int intToStr(int x, char str[], int d);
void ftoa(float n, char *res, int afterpoint);
void bubble_sort_top(float *arr, int len);
// ===========================================================
// utils

void add_bias_TA(float *output, float *biases, int batch, int n, int size);
void scale_bias_TA(float *output, float *scales, int batch, int n, int size);

void fill_cpu_TA(int N, float ALPHA, float *X, int INCX);
void fill_cpu_zero_TA(float *X, int N);
// int uint32ToInt(uint32_t val);



#endif