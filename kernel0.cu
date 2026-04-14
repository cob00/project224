#include "common.h"
#include "matrix.h"

__global__ void sptrsv_kernel0(
    unsigned int  numRows,
    unsigned int* rowPtrs,
    unsigned int* colIdxs,
    float*        matValues,
    unsigned int* cscColPtrs,
    unsigned int* cscRowIdxs,
    float*        bValues,
    float*        xValues,
    unsigned int  numCols,
    int*          dep_counter,
    unsigned int* row_dep_count
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (row >= numRows || col >= numCols) return;

    if (row > 0) {
        while (atomicAdd(&dep_counter[row], 0) < (int)row_dep_count[row]) { 
            __nanosleep(10);
        }
    }
    __threadfence();

    unsigned int rowStart = rowPtrs[row];
    unsigned int rowEnd   = rowPtrs[row + 1];

    float diag = 1.0f;
    for (unsigned int j = rowStart; j < rowEnd; ++j) {
        if (colIdxs[j] == row) {
            diag = matValues[j] != 0.0f ? matValues[j] : 1.0f;
            break;
        }
    }

    float sum = bValues[row * numCols + col];
    for (unsigned int j = rowStart; j < rowEnd; ++j) {
        unsigned int c = colIdxs[j];
        if (c < row) {
            sum -= matValues[j] * xValues[c * numCols + col];
        }
    }

    xValues[row * numCols + col] = sum / diag;

    __syncthreads();

    if (col == 0) {
        __threadfence();
        unsigned int cStart = cscColPtrs[row];
        unsigned int cEnd   = cscColPtrs[row + 1];
        for (unsigned int j = cStart; j < cEnd; ++j) {
            unsigned int dep = cscRowIdxs[j];
            if (dep > row) {
                atomicAdd(&dep_counter[dep], 1);
            }
        }
    }
}


void sptrsv_gpu0(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    unsigned int* row_dep_count_h = (unsigned int*)calloc(n, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int count = 0;
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            if (L_r_host->colIdxs[j] < i) count++;
        }
        row_dep_count_h[i] = count;
    }

    unsigned int* row_dep_count_d;
    int*          dep_counter_d;
    cudaMalloc(&row_dep_count_d, n * sizeof(unsigned int));
    cudaMalloc(&dep_counter_d,   n * sizeof(int));

    cudaMemcpy(row_dep_count_d, row_dep_count_h, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(dep_counter_d, 0, n * sizeof(int));

    free(row_dep_count_h);

    CSRMatrix csr_shadow;
    cudaMemcpy(&csr_shadow, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);

    CSCMatrix csc_shadow;
    cudaMemcpy(&csc_shadow, L_c, sizeof(CSCMatrix), cudaMemcpyDeviceToHost);

    DenseMatrix b_shadow, x_shadow;
    cudaMemcpy(&b_shadow, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x_shadow, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    dim3 grid(n);
    dim3 block(numCols);

    sptrsv_kernel0<<<grid, block>>>(
        n,
        csr_shadow.rowPtrs,
        csr_shadow.colIdxs,
        csr_shadow.values,
        csc_shadow.colPtrs,
        csc_shadow.rowIdxs,
        b_shadow.values,
        x_shadow.values,
        numCols,
        dep_counter_d,
        row_dep_count_d
    );

    cudaFree(dep_counter_d);
    cudaFree(row_dep_count_d);
}