#include "gemm.h"

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {  
#endif

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc)
{
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#if OPENMP
// #pragma omp parallel for
#pragma omp parallel for private(i, j, k) shared(A, B, C, ALPHA, M, N, K, lda, ldb, ldc)
#endif
    for(i = 0; i < M; ++i){ // M : 卷积核数量
        for(k = 0; k < K; ++k){ // size * size
            // register 是一个存储类说明符
            // 表示该变量 A_PART 应该尽可能地存储在 CPU 的寄存器中 而不是在内存中
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{

    //printf("1=%d,2=%d,3=%d,ALPHA=%d,1=%d,2=%d,3=%d\n",M,K,N,ALPHA,lda,ldb,ldc);
    //debug_plot("A",A, K*M);
    //debug_plot("B",B, K*N);
    ///debug_plot("C",C, N*M);

    //printf("stoping");

    int i,j,k;
#if OPENMP
// #pragma omp parallel for
#pragma omp parallel for private(j, k) shared(A, B, C, ALPHA, M, N, K, lda, ldb, ldc)
#endif
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            //printf("sum");
            for(k = 0; k < K; ++k){
                //if(k % 1 == 0 & roundnum == 2){
                    //printf("+ %f",ALPHA*A[i*lda+k]*B[j*ldb + k]);
                    //printf("1=%f,2=%d,3=%f,4=%d,5=%f\n",ALPHA*A[i*lda+k]*B[j*ldb + k], i*lda+k, A[i*lda+k], j*ldb + k, B[j*ldb + k]);
                //}
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            //printf("\n");
            //printf("j=%d,sum=%f\n",j,sum);
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#if OPENMP
// #pragma omp parallel for
#endif
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
            //printf("M=%d,N=%d,K=%d,lda=%d,ldb=%d,ldc=%d\n",M,N,K,lda,ldb,ldc);
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#if OPENMP
// #pragma omp parallel for
#endif
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    // 矩阵乘 TA TB 控制是否进行矩阵转制
    // ALPHA 缩放因子
    
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }

    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}


#ifdef __cplusplus
}
#endif
