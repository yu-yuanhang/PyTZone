#include "all.h"


void add_bias_TA(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}
void scale_bias_TA(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}


void fill_cpu_TA(int N, float ALPHA, float *X, int INCX) {
    for(int i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void fill_cpu_zero_TA(float *X, int N) {
    memset(X, 0, N * sizeof(float));
}

// int uint32ToInt(uint32_t val) {
//     // return -1 when error
//     ERROR_CHECK_RET_TA(val > INT32_MAX, true, "Overflow when converting uint32_t to int32");
//     return (int)val;
// }