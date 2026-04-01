#include "common.h"

void sptrsv_cpu(CSRMatrix* L, DenseMatrix* B, DenseMatrix* X){

    unsigned int n = L->numRows;
    unsigned int nB = B->numCols;

    for(unsigned int b = 0; b < nB; ++b){
        for(unsigned int i = 0; i < n; ++i){
            float sum = B->values[i * nB + b];
            float diag = 0.0f;

            for(unsigned int idx = L->rowPtrs[i]; idx < L->rowPtrs[i + 1]; ++idx){
                unsigned int col = L->colIdxs[idx];
                float val = L->values[idx];
                if(col < i){
                    sum -= val * X->values[col * nB + b];
                } else if(col == i){
                    diag = val !=0 ? val : 1.0f; // Avoid division by zero
                }
            }
            X->values[i * nB + b] = sum / diag;
        }
    }
}