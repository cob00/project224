
#ifndef _COMMON_H_
#define _COMMON_H_

#include "matrix.h"

#define CUDA_ERROR_CHECK(call)                                                                              \
    {                                                                                                       \
        cudaError_t err = call;                                                                             \
        if(err != cudaSuccess) {                                                                            \
            fprintf(stderr, "CUDA Error: CUDA call \"%s\" on line %d in file %s failed with %s (%d).\n",    \
                    #call, __LINE__, __FILE__, cudaGetErrorString(err), err);                               \
            exit(err);                                                                                      \
        }                                                                                                   \
    }

void sptrsv_cpu(CSRMatrix* L, DenseMatrix* B, DenseMatrix* X);
void sptrsv_gpu0(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols);
void sptrsv_gpu1(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols);
void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols);
void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols);
/*void spmspm_cpu1();
void spmspm_gpu0();
void spmspm_gpu1();
void spmspm_gpu2();
void spmspm_gpu3();
void spmspm_gpu4();
void spmspm_gpu5();
void spmspm_gpu6();
void spmspm_gpu7();
void spmspm_gpu8();
void spmspm_gpu9();
*/
#endif

