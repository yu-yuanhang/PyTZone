#include "math.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {  
#endif

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
// #if OPENMP
// #pragma omp parallel for
// #endif
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
// #if OPENMP
// #pragma omp parallel for
// #endif
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
// #if OPENMP
// #pragma omp parallel for
// #endif
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(variance[f]);
                x[index] = (x[index] - mean[f])/(ca_sqrt(variance[f]) + .000001f);
            }
        }
    }
}


void fill_cpu(int N, float ALPHA, float *X, int INCX) {
    for(int i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void fill_cpu_zero(float *X, int N) {
    memset(X, 0, N * sizeof(float));
}

// int uint32ToInt(uint32_t val) {
//     // return -1 when error
//     ERROR_CHECK_RET_TA(val > INT32_MAX, true, "Overflow when converting uint32_t to int32");
//     return (int)val;
// }



#ifdef __cplusplus
}
#endif
