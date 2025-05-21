#ifndef __MATH_H__
#define __MATH_H__

#ifdef __cplusplus
extern "C" {  
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* math.c */
#define PI 3.14159265358979323846
#define e  2.7182818284590452354
#define ln_2 0.69314718055994530942
#define ln_10 2.30258509299404568402
#define first_aim_money 1000000000.0f

#define fabs(a) ((a)>0?(a):(-(a)))

float ca_max(float a, float b);
double ca_pow(double a,int n);
double ca_eee(double x);
double ca_exp(double x);
// float ca_rand();
int ca_floor(double x);
double ca_sqrt(double x);
double ca_ln(double x);
double ca_log(double a,double N);
double ca_sin(double x);
double ca_cos(double x);
double can(double x);

void reverse(char *str, int len);
int intToStr(int x, char str[], int d);
void ftoa(float n, char *res, int afterpoint);
void bubble_sort_top(float *arr, int len);
// ===========================================================
// utils

void add_bias(float *output, float *biases, int batch, int n, int size);
void scale_bias(float *output, float *scales, int batch, int n, int size);

void fill_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu_zero(float *X, int N);
// int uint32ToInt(uint32_t val);

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
// ===========================================================
// activate
static inline float logistic_activate(float x){return 1./(1. + ca_exp(-x));}
static inline float relu_activate(float x){return x*(x>0);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float linear_activate(float x){return x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float tanh_activate(float x){return (ca_exp(2*x)-1)/(ca_exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}
static inline float leaky_activate(float x){return (x>0) ? x : (1e-2)*x;}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(ca_exp(x)-1);}
static inline float loggy_activate(float x){return 2./(1. + ca_exp(-x)) - 1;}
static inline float stair_activate(float x)
{
    int n = ca_floor(x);
    if (n%2 == 0) return ca_floor(x/2.);
    else return (x - n) + ca_floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(ca_exp(x)-1);}

#ifdef __cplusplus
}
#endif

#endif 