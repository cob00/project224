#include "common.h"
#include "matrix.h"

#define TILE 512
#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 128
#endif

#define PCR_BLOCK_SIZE 256

// ============================================================================
// PCR Kernel 1: Initialize coefficients
// X[i] = a[i] * X[i-1] + c[i,col]
// where: a[i] = -L[i,i-1] / L[i,i], c[i,col] = B[i,col] / L[i,i]
// ============================================================================
__global__ void pcr_init(
    unsigned int  numRows,
    unsigned int* rowPtrs,
    unsigned int* colIdxs,
    float*        matValues,
    float*        bValues,
    unsigned int  numCols,
    float*        a_coeff,
    float*        c_coeff
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows) return;
    
    float left = 0.0f, diag = 1.0f;
    
    for (unsigned int j = rowPtrs[row]; j < rowPtrs[row + 1]; ++j) {
        unsigned int c = colIdxs[j];
        if ((int)c == row - 1) left = matValues[j];
        else if ((int)c == row) diag = matValues[j];
    }
    
    a_coeff[row] = (row > 0 && diag != 0.0f) ? (-left / diag) : 0.0f;
    
    float diag_inv = (diag != 0.0f) ? (1.0f / diag) : 1.0f;
    for (unsigned int col = 0; col < numCols; col++) {
        c_coeff[row * numCols + col] = bValues[row * numCols + col] * diag_inv;
    }
}

// ============================================================================
// PCR Kernel 2: One PCR step - eliminate dependency at 'offset' distance
// After: X[i] depends on X[i-2*offset] instead of X[i-offset]
// ============================================================================
__global__ void pcr_step(
    unsigned int numRows,
    int offset,
    float* a_in,
    float* c_in,
    float* a_out,
    float* c_out,
    unsigned int numCols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows) return;
    
    int prev_row = row - offset;
    float a_curr = a_in[row];
    
    if (prev_row < 0) {
        a_out[row] = 0.0f;
        for (unsigned int col = 0; col < numCols; col++) {
            c_out[row * numCols + col] = c_in[row * numCols + col];
        }
    } else {
        float a_prev = a_in[prev_row];
        a_out[row] = a_curr * a_prev;
        for (unsigned int col = 0; col < numCols; col++) {
            float c_curr = c_in[row * numCols + col];
            float c_prev = c_in[prev_row * numCols + col];
            c_out[row * numCols + col] = a_curr * c_prev + c_curr;
        }
    }
}

// ============================================================================
// PCR Kernel 3: Write final result. After PCR, X[i] = c[i]
// ============================================================================
__global__ void pcr_finalize(
    unsigned int numRows,
    unsigned int numCols,
    float* c_coeff,
    float* xValues
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows) return;
    
    for (unsigned int col = 0; col < numCols; col++) {
        xValues[row * numCols + col] = c_coeff[row * numCols + col];
    }
}

// ============================================================================
// FALLBACK: Same as kernel 2 (for non-tridiagonal matrices)
// ============================================================================
__global__ void kernel3_fallback(
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
                while (atomicAdd(&dep_counter[row], 0) < (int)row_dep_count[row]) {
                    __nanosleep(10);
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
                    sum -= val * xValues[c * numCols + col];
                } else if ((int)c == row) {
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
                if ((int)dep > row) atomicAdd(&dep_counter[dep], 1);
            }
            s_ready[r] = 1;
        }
    }
}

// ============================================================================
// HOST: Detect tridiagonal, dispatch to PCR or fallback
// ============================================================================
void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    // *** Detect if matrix is strictly tridiagonal ***
    bool is_tridiagonal = true;
    for (unsigned int i = 0; i < n && is_tridiagonal; ++i) {
        unsigned int dep_count = 0;
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            unsigned int c = L_r_host->colIdxs[j];
            if (c < i) {
                dep_count++;
                if (c != i - 1) { is_tridiagonal = false; break; }
            }
        }
        if (dep_count > 1) is_tridiagonal = false;
    }

    CSRMatrix csr_shadow;
    cudaMemcpy(&csr_shadow, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix b_shadow, x_shadow;
    cudaMemcpy(&b_shadow, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x_shadow, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    if (is_tridiagonal) {
        // ====================================================================
        // PCR PATH: O(log n) parallel solve
        // ====================================================================
        printf("  [PCR mode: matrix is tridiagonal]\n");

        float *a_A, *a_B, *c_A, *c_B;
        cudaMalloc(&a_A, n * sizeof(float));
        cudaMalloc(&a_B, n * sizeof(float));
        cudaMalloc(&c_A, n * numCols * sizeof(float));
        cudaMalloc(&c_B, n * numCols * sizeof(float));

        dim3 block(PCR_BLOCK_SIZE);
        dim3 grid((n + PCR_BLOCK_SIZE - 1) / PCR_BLOCK_SIZE);

        // Initialize coefficients
        pcr_init<<<grid, block>>>(
            n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
            b_shadow.values, numCols, a_A, c_A
        );

        // Apply log(n) PCR steps with double buffering
        float *a_in = a_A, *c_in = c_A;
        float *a_out = a_B, *c_out = c_B;
        
        int offset = 1;
        while (offset < (int)n) {
            pcr_step<<<grid, block>>>(n, offset, a_in, c_in, a_out, c_out, numCols);
            float *tmp = a_in; a_in = a_out; a_out = tmp;
            tmp = c_in; c_in = c_out; c_out = tmp;
            offset <<= 1;
        }

        // Write final result
        pcr_finalize<<<grid, block>>>(n, numCols, c_in, x_shadow.values);

        cudaFree(a_A); cudaFree(a_B);
        cudaFree(c_A); cudaFree(c_B);

    } else {
        // ====================================================================
        // FALLBACK PATH: Kernel 2 logic
        // ====================================================================
        printf("  [Fallback mode: matrix is not tridiagonal]\n");

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

        CSCMatrix csc_shadow;
        cudaMemcpy(&csc_shadow, L_c, sizeof(CSCMatrix), cudaMemcpyDeviceToHost);

        unsigned int smemSize = TILE * (sizeof(unsigned int) + sizeof(float)) + rpb * sizeof(int);
        dim3 grid((n + rpb - 1) / rpb);
        dim3 block(numCols);

        kernel3_fallback<<<grid, block, smemSize>>>(
            n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
            csc_shadow.colPtrs, csc_shadow.rowIdxs, b_shadow.values, x_shadow.values,
            numCols, dep_counter_d, row_dep_count_d, rpb
        );

        cudaFree(dep_counter_d);
        cudaFree(row_dep_count_d);
    }
}
