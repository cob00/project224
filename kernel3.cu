#include <stdio.h>
#include "common.h"
#include "matrix.h"

#define TILE 512
#define MAX_REFINE_ITERS 3

#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 128
#endif

// 2D PCR Init - coalesced memory access (threadIdx.x = col, threadIdx.y = row in block)
__global__ void pcr_init_2d(
    unsigned int numRows, unsigned int* rowPtrs, unsigned int* colIdxs, float* matValues,
    float* rhs, unsigned int numCols, float* a_coeff, float* c_coeff
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows || col >= (int)numCols) return;
    
    // Find left and diag (each thread does this; cheap since row entries are few)
    float left = 0.0f, diag = 1.0f;
    for (unsigned int j = rowPtrs[row]; j < rowPtrs[row + 1]; ++j) {
        unsigned int c = colIdxs[j];
        if ((int)c == row - 1) left = matValues[j];
        else if ((int)c == row) diag = matValues[j];
    }
    
    if (col == 0) a_coeff[row] = (row > 0 && diag != 0.0f) ? (-left / diag) : 0.0f;
    
    float diag_inv = (diag != 0.0f) ? (1.0f / diag) : 1.0f;
    c_coeff[row * numCols + col] = rhs[row * numCols + col] * diag_inv;
}

// 2D PCR Step - coalesced
__global__ void pcr_step_2d(
    unsigned int numRows, int offset, unsigned int numCols,
    float* a_in, float* c_in, float* a_out, float* c_out
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows || col >= (int)numCols) return;
    
    int prev_row = row - offset;
    if (prev_row < 0) {
        c_out[row * numCols + col] = c_in[row * numCols + col];
        if (col == 0) a_out[row] = 0.0f;
    } else {
        float a_curr = a_in[row];
        float c_curr = c_in[row * numCols + col];
        float c_prev = c_in[prev_row * numCols + col];
        c_out[row * numCols + col] = a_curr * c_prev + c_curr;
        if (col == 0) a_out[row] = a_curr * a_in[prev_row];
    }
}

// 2D residual - coalesced
__global__ void compute_residual_2d(
    unsigned int numRows, unsigned int* rowPtrs, unsigned int* colIdxs, float* matValues,
    float* X, float* B, float* R, unsigned int numCols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows || col >= (int)numCols) return;
    
    float sum = B[row * numCols + col];
    for (unsigned int j = rowPtrs[row]; j < rowPtrs[row + 1]; ++j) {
        sum -= matValues[j] * X[colIdxs[j] * numCols + col];
    }
    R[row * numCols + col] = sum;
}

// 2D update solution
__global__ void update_solution_2d(float* X, float* dX, unsigned int numRows, unsigned int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows || col >= (int)numCols) return;
    X[row * numCols + col] += dX[row * numCols + col];
}

// 2D finalize
__global__ void pcr_finalize_2d(unsigned int numRows, unsigned int numCols, float* c_coeff, float* xValues) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows || col >= (int)numCols) return;
    xValues[row * numCols + col] = c_coeff[row * numCols + col];
}

// Fallback (kernel 2 logic)
__global__ void kernel3_fallback(
    unsigned int numRows, unsigned int* rowPtrs, unsigned int* colIdxs, float* matValues,
    unsigned int* cscColPtrs, unsigned int* cscRowIdxs,
    float* bValues, float* xValues, unsigned int numCols,
    int* dep_counter, unsigned int* row_dep_count, int rows_per_block
) {
    extern __shared__ char smem[];
    unsigned int* s_cols = (unsigned int*) smem;
    float* s_vals = (float*)(smem + TILE * sizeof(unsigned int));
    volatile int* s_ready = (volatile int*)(smem + TILE * sizeof(unsigned int) + TILE * sizeof(float));
    unsigned int col = threadIdx.x;
    int first_row = blockIdx.x * rows_per_block;
    for (int i = col; i < rows_per_block; i += numCols) s_ready[i] = 0;
    __syncthreads();
    for (int r = 0; r < rows_per_block; r++) {
        int row = first_row + r;
        if (row >= (int)numRows) break;
        if (col == 0) {
            if (r > 0) while (s_ready[r - 1] == 0) {}
            if (row > 0) while (atomicAdd(&dep_counter[row], 0) < (int)row_dep_count[row]) __nanosleep(10);
        }
        __syncthreads();
        __threadfence();
        unsigned int rowStart = rowPtrs[row], rowEnd = rowPtrs[row + 1];
        float sum = bValues[row * numCols + col], diag = 1.0f;
        for (unsigned int base = rowStart; base < rowEnd; base += TILE) {
            unsigned int tileEnd = (base + TILE < rowEnd) ? base + TILE : rowEnd;
            unsigned int tileSize = tileEnd - base;
            for (unsigned int j = col; j < tileSize; j += numCols) {
                s_cols[j] = colIdxs[base + j];
                s_vals[j] = matValues[base + j];
            }
            __syncthreads();
            for (unsigned int j = 0; j < tileSize; ++j) {
                unsigned int c = s_cols[j];
                float val = s_vals[j];
                if ((int)c < row) sum -= val * xValues[c * numCols + col];
                else if ((int)c == row) diag = val != 0.0f ? val : 1.0f;
            }
            __syncthreads();
        }
        xValues[row * numCols + col] = sum / diag;
        __syncthreads();
        if (col == 0) {
            __threadfence();
            unsigned int cStart = cscColPtrs[row], cEnd = cscColPtrs[row + 1];
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

    // Detect if matrix has L[i,i-1] for most rows
    int has_prev_count = 0;
    for (unsigned int i = 1; i < n; ++i) {
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            if (L_r_host->colIdxs[j] == i - 1) { has_prev_count++; break; }
        }
    }
    float ratio = (n > 1) ? (float)has_prev_count / (n - 1) : 0.0f;

    CSRMatrix csr_shadow;
    cudaMemcpy(&csr_shadow, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix b_shadow, x_shadow;
    cudaMemcpy(&b_shadow, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x_shadow, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    if (ratio > 0.95f) {
        printf("  [PCR + refinement (2D coalesced), %d iters]\n", MAX_REFINE_ITERS);
        size_t a_size = n * sizeof(float);
        size_t c_size = (size_t)n * numCols * sizeof(float);
        float *a_A, *a_B, *c_A, *c_B, *residual;
        cudaMalloc(&a_A, a_size); cudaMalloc(&a_B, a_size);
        cudaMalloc(&c_A, c_size); cudaMalloc(&c_B, c_size);
        cudaMalloc(&residual, c_size);

        // 2D launch config (block.y=16 to keep grid.y < 65535 limit)
        dim3 block2d(32, 16);  // 512 threads per block
        dim3 grid2d((numCols + 31) / 32, (n + 15) / 16);

        // Initial PCR using B
        pcr_init_2d<<<grid2d, block2d>>>(n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
                                          b_shadow.values, numCols, a_A, c_A);

        float *a_in = a_A, *c_in = c_A, *a_out = a_B, *c_out = c_B;
        int offset = 1;
        while (offset < (int)n) {
            pcr_step_2d<<<grid2d, block2d>>>(n, offset, numCols, a_in, c_in, a_out, c_out);
            float *t = a_in; a_in = a_out; a_out = t;
            t = c_in; c_in = c_out; c_out = t;
            offset <<= 1;
        }
        pcr_finalize_2d<<<grid2d, block2d>>>(n, numCols, c_in, x_shadow.values);

        // Iterative refinement
        for (int iter = 0; iter < MAX_REFINE_ITERS; iter++) {
            compute_residual_2d<<<grid2d, block2d>>>(n, csr_shadow.rowPtrs, csr_shadow.colIdxs,
                                                     csr_shadow.values, x_shadow.values, b_shadow.values, residual, numCols);
            
            pcr_init_2d<<<grid2d, block2d>>>(n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
                                              residual, numCols, a_A, c_A);
            a_in = a_A; c_in = c_A; a_out = a_B; c_out = c_B;
            offset = 1;
            while (offset < (int)n) {
                pcr_step_2d<<<grid2d, block2d>>>(n, offset, numCols, a_in, c_in, a_out, c_out);
                float *t = a_in; a_in = a_out; a_out = t;
                t = c_in; c_in = c_out; c_out = t;
                offset <<= 1;
            }
            update_solution_2d<<<grid2d, block2d>>>(x_shadow.values, c_in, n, numCols);
        }

        cudaFree(a_A); cudaFree(a_B); cudaFree(c_A); cudaFree(c_B); cudaFree(residual);
    } else {
        printf("  [Fallback]\n");
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
        dim3 grid((n + rpb - 1) / rpb), block(numCols);
        kernel3_fallback<<<grid, block, smemSize>>>(n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
            csc_shadow.colPtrs, csc_shadow.rowIdxs, b_shadow.values, x_shadow.values,
            numCols, dep_counter_d, row_dep_count_d, rpb);
        cudaFree(dep_counter_d); cudaFree(row_dep_count_d);
    }
}
