#include "common.h"
#include "matrix.h"

#define TILE 512

__global__ void sptrsv_kernel1(
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

    extern __shared__ char smem[];
    unsigned int* s_cols = (unsigned int*) smem;
    float*        s_vals = (float*)(smem + TILE * sizeof(unsigned int));

    float sum  = bValues[row * numCols + col];
    float diag = 1.0f;

    for (unsigned int base = rowStart; base < rowEnd; base += TILE) {
        unsigned int tileEnd  = base + TILE < rowEnd ? base + TILE : rowEnd;
        unsigned int tileSize = tileEnd - base;

        // Collaboratively load tile into shared memory
        for (unsigned int j = col; j < tileSize; j += numCols) {
            s_cols[j] = colIdxs[base + j];
            s_vals[j] = matValues[base + j];
        }
        __syncthreads();

        // All threads compute using shared tile
        for (unsigned int j = 0; j < tileSize; ++j) {
            unsigned int c = s_cols[j];
            float val = s_vals[j];
            if (c < row) {
                sum -= val * xValues[c * numCols + col];
            } else if (c == row) {
                diag = val != 0.0f ? val : 1.0f;
            }
        }
        __syncthreads();
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


void sptrsv_gpu1(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
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

    // Shared memory: TILE * (uint + float) = 512 * 8 = 4096 bytes — well within 48KB
    unsigned int smemSize = TILE * (sizeof(unsigned int) + sizeof(float));

    dim3 grid(n);
    dim3 block(numCols);

    sptrsv_kernel1<<<grid, block, smemSize>>>(
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