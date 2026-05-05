#include "common.h"
#include "matrix.h"

#define TILE 512
#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 128
#endif

// ============================================================================
// OPTIMIZATION: Shared Memory Cache for In-Block Chain Dependencies
// 
// For chain-like matrices (tmt_sym), each row reads X[i-1] from global memory
// (latency ~200ns per access). By caching the previous row's X values in shared
// memory (latency ~30ns), we eliminate the global memory dependency for the
// dominant chain access pattern.
// 
// For 726k rows × 512 cols, this eliminates ~370M global memory reads,
// replacing them with ~370M shared memory reads (6-8x faster).
// ============================================================================

__global__ void sptrsv_kernel3(
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
    unsigned int* row_dep_count,
    int           rows_per_block
) {
    extern __shared__ char smem[];
    unsigned int* s_cols  = (unsigned int*) smem;
    float*        s_vals  = (float*)(smem + TILE * sizeof(unsigned int));
    volatile int* s_ready = (volatile int*)(smem + TILE * sizeof(unsigned int) + TILE * sizeof(float));
    // *** NEW: Shared memory cache for previous row's X values ***
    float* s_X_prev = (float*)(smem + TILE * sizeof(unsigned int) + TILE * sizeof(float) + rows_per_block * sizeof(int));

    unsigned int col       = threadIdx.x;
    int          first_row = blockIdx.x * rows_per_block;

    for (int i = col; i < rows_per_block; i += numCols) s_ready[i] = 0;
    __syncthreads();

    for (int r = 0; r < rows_per_block; r++) {
        int row = first_row + r;
        if (row >= (int)numRows) break;

        if (col == 0) {
            if (r > 0) {
                while (s_ready[r - 1] == 0) {}
            }
            if (row > 0) {
                int target = (int)row_dep_count[row];
                int local_counter = atomicAdd(&dep_counter[row], 0);
                int wait_ns = 32;
                while (local_counter < target) {
                    __nanosleep(wait_ns);
                    local_counter = atomicAdd(&dep_counter[row], 0);
                    if (wait_ns < 2048) wait_ns <<= 1;
                }
            }
        }
        __syncthreads();
        __threadfence();

        unsigned int rowStart = rowPtrs[row];
        unsigned int rowEnd   = rowPtrs[row + 1];

        float sum  = bValues[row * numCols + col];
        float diag = 1.0f;

        for (unsigned int base = rowStart; base < rowEnd; base += TILE) {
            unsigned int tileEnd  = (base + TILE < rowEnd) ? base + TILE : rowEnd;
            unsigned int tileSize = tileEnd - base;

            for (unsigned int j = col; j < tileSize; j += numCols) {
                s_cols[j] = colIdxs[base + j];
                s_vals[j] = matValues[base + j];
            }
            __syncthreads();

            for (unsigned int j = 0; j < tileSize; ++j) {
                unsigned int c = s_cols[j];
                float val = s_vals[j];
                if ((int)c < row) {
                    // *** OPTIMIZATION: Use shared memory cache for chain dep ***
                    if ((int)c == row - 1 && r > 0) {
                        sum -= val * s_X_prev[col];  // FAST: shared memory read
                    } else {
                        sum -= val * xValues[c * numCols + col];  // Fallback: global memory
                    }
                } else if ((int)c == row) {
                    diag = val != 0.0f ? val : 1.0f;
                }
            }
            __syncthreads();
        }

        // Compute and write result
        float result = sum / diag;
        xValues[row * numCols + col] = result;
        s_X_prev[col] = result;  // *** Cache for next row's chain dep ***
        __syncthreads();

        if (col == 0) {
            __threadfence();
            unsigned int cStart = cscColPtrs[row];
            unsigned int cEnd   = cscColPtrs[row + 1];
            for (unsigned int j = cStart; j < cEnd; ++j) {
                unsigned int dep = cscRowIdxs[j];
                if ((int)dep > row) atomicAdd(&dep_counter[dep], 1);
            }
            s_ready[r] = 1;
        }
    }
}

void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    unsigned int* row_dep_count_h = (unsigned int*)calloc(n, sizeof(unsigned int));
    unsigned int sequential_count = 0;

    for (unsigned int i = 0; i < n; ++i) {
        unsigned int count = 0;
        bool depends_on_prev = false;
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            if (L_r_host->colIdxs[j] < i) {
                count++;
                if (L_r_host->colIdxs[j] == i - 1) depends_on_prev = true;
            }
        }
        row_dep_count_h[i] = count;
        if (depends_on_prev) sequential_count++;
    }

    float seq_ratio = (n > 1) ? (float)sequential_count / (float)(n - 1) : 0.0f;
    int rpb = (seq_ratio >= 0.5f) ? 128 : 1;

    unsigned int* row_dep_count_d;
    int* dep_counter_d;
    cudaMalloc(&row_dep_count_d, n * sizeof(unsigned int));
    cudaMalloc(&dep_counter_d, n * sizeof(int));
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

    // *** UPDATED: Add shared memory for s_X_prev cache ***
    unsigned int smemSize = TILE * (sizeof(unsigned int) + sizeof(float)) 
                          + rpb * sizeof(int) 
                          + numCols * sizeof(float);  // s_X_prev cache
    
    dim3 grid((n + rpb - 1) / rpb);
    dim3 block(numCols);

    sptrsv_kernel3<<<grid, block, smemSize>>>(
        n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
        csc_shadow.colPtrs, csc_shadow.rowIdxs, b_shadow.values, x_shadow.values,
        numCols, dep_counter_d, row_dep_count_d, rpb
    );

    cudaFree(dep_counter_d);
    cudaFree(row_dep_count_d);
}
