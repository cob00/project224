#include <stdio.h>
#include "common.h"
#include "matrix.h"

#define TILE 512
#define BPCR_BLOCK 8           // Block size for banded PCR (must be > max_dep)
#define PCR_THREADS 256

#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 128
#endif

// ============================================================================
// Banded PCR Helper: Build A[I] (8x8) and C[I] (8 x numCols) for each block
// 
// Block-bidiagonal form:  D[I] * X[I] + S[I] * X[I-1] = B[I]
// Solve for: X[I] = -D[I]^-1 * S[I] * X[I-1] + D[I]^-1 * B[I] = A[I] * X[I-1] + C[I]
// ============================================================================
__global__ void bpcr_init(
    unsigned int numRows,
    unsigned int* rowPtrs,
    unsigned int* colIdxs,
    float*        matValues,
    float*        bValues,
    unsigned int  numCols,
    float*        A,    // [N_blocks * 8 * 8]
    float*        C     // [N_blocks * 8 * numCols]
) {
    int I = blockIdx.x;                 // Block index (one block per CUDA block)
    int col = threadIdx.x;              // Each thread handles one RHS column
    
    int row_start = I * BPCR_BLOCK;
    int N_blocks = (numRows + BPCR_BLOCK - 1) / BPCR_BLOCK;
    if (I >= N_blocks) return;

    __shared__ float D[BPCR_BLOCK][BPCR_BLOCK];   // Diagonal block (lower triangular)
    __shared__ float S[BPCR_BLOCK][BPCR_BLOCK];   // Subdiagonal block
    __shared__ float Atemp[BPCR_BLOCK][BPCR_BLOCK]; // Temp for A computation
    
    // Initialize D and S to identity / zero
    if (col < BPCR_BLOCK) {
        for (int j = 0; j < BPCR_BLOCK; j++) {
            D[col][j] = (col == j) ? 1.0f : 0.0f;
            S[col][j] = 0.0f;
        }
    }
    __syncthreads();

    // Thread 0 builds D and S from L matrix
    if (col == 0) {
        for (int k = 0; k < BPCR_BLOCK; k++) {
            int row = row_start + k;
            if (row >= (int)numRows) break;
            
            for (unsigned int j = rowPtrs[row]; j < rowPtrs[row + 1]; j++) {
                int c = (int)colIdxs[j];
                float val = matValues[j];
                
                if (c >= row_start && c <= row) {
                    // Within current block (lower triangular part)
                    D[k][c - row_start] = val;
                } else if (c >= (row_start - BPCR_BLOCK) && c < row_start) {
                    // Previous block
                    S[k][c - (row_start - BPCR_BLOCK)] = val;
                }
            }
        }
    }
    __syncthreads();

    // Solve D * Atemp = -S (column by column) - thread 0 only
    if (col == 0) {
        for (int j = 0; j < BPCR_BLOCK; j++) {
            for (int k = 0; k < BPCR_BLOCK; k++) {
                float rhs = -S[k][j];
                for (int m = 0; m < k; m++) {
                    rhs -= D[k][m] * Atemp[m][j];
                }
                Atemp[k][j] = (D[k][k] != 0.0f) ? rhs / D[k][k] : 0.0f;
            }
        }
        // Write A[I]
        for (int k = 0; k < BPCR_BLOCK; k++) {
            for (int j = 0; j < BPCR_BLOCK; j++) {
                A[I * BPCR_BLOCK * BPCR_BLOCK + k * BPCR_BLOCK + j] = Atemp[k][j];
            }
        }
    }
    __syncthreads();

    // Solve D * C[:,col] = B[:,col] for each thread's column
    if (col < (int)numCols) {
        float c_local[BPCR_BLOCK];
        for (int k = 0; k < BPCR_BLOCK; k++) {
            int row = row_start + k;
            float rhs = (row < (int)numRows) ? bValues[row * numCols + col] : 0.0f;
            for (int m = 0; m < k; m++) {
                rhs -= D[k][m] * c_local[m];
            }
            c_local[k] = (D[k][k] != 0.0f) ? rhs / D[k][k] : 0.0f;
        }
        // Write C[I][:,col]
        for (int k = 0; k < BPCR_BLOCK; k++) {
            C[(I * BPCR_BLOCK + k) * numCols + col] = c_local[k];
        }
    }
}

// ============================================================================
// Banded PCR Step: A_new[I] = A[I] * A[I-offset], C_new[I] = A[I] * C[I-offset] + C[I]
// ============================================================================
__global__ void bpcr_step(
    unsigned int N_blocks,
    int offset,
    float* A_in, float* C_in,
    float* A_out, float* C_out,
    unsigned int numCols
) {
    int I = blockIdx.x;
    int col = threadIdx.x;
    int prev_I = I - offset;
    
    if (I >= (int)N_blocks) return;

    __shared__ float A_curr[BPCR_BLOCK][BPCR_BLOCK];
    __shared__ float A_prev[BPCR_BLOCK][BPCR_BLOCK];
    
    // Load A[I] and A[I-offset] into shared
    if (col < BPCR_BLOCK * BPCR_BLOCK) {
        int k = col / BPCR_BLOCK;
        int j = col % BPCR_BLOCK;
        A_curr[k][j] = A_in[I * BPCR_BLOCK * BPCR_BLOCK + k * BPCR_BLOCK + j];
        if (prev_I >= 0) {
            A_prev[k][j] = A_in[prev_I * BPCR_BLOCK * BPCR_BLOCK + k * BPCR_BLOCK + j];
        } else {
            A_prev[k][j] = 0.0f;
        }
    }
    __syncthreads();

    // A_out[I] = A_curr * A_prev (8x8 matrix multiply)
    if (col == 0) {
        for (int k = 0; k < BPCR_BLOCK; k++) {
            for (int j = 0; j < BPCR_BLOCK; j++) {
                float sum = 0.0f;
                for (int m = 0; m < BPCR_BLOCK; m++) {
                    sum += A_curr[k][m] * A_prev[m][j];
                }
                A_out[I * BPCR_BLOCK * BPCR_BLOCK + k * BPCR_BLOCK + j] = sum;
            }
        }
    }

    // C_out[I][:,col] = A_curr * C_prev[:,col] + C_curr[:,col]
    if (col < (int)numCols) {
        float C_prev_col[BPCR_BLOCK];
        for (int k = 0; k < BPCR_BLOCK; k++) {
            if (prev_I >= 0) {
                C_prev_col[k] = C_in[(prev_I * BPCR_BLOCK + k) * numCols + col];
            } else {
                C_prev_col[k] = 0.0f;
            }
        }
        
        for (int k = 0; k < BPCR_BLOCK; k++) {
            float sum = C_in[(I * BPCR_BLOCK + k) * numCols + col];
            for (int m = 0; m < BPCR_BLOCK; m++) {
                sum += A_curr[k][m] * C_prev_col[m];
            }
            C_out[(I * BPCR_BLOCK + k) * numCols + col] = sum;
        }
    }
}

// ============================================================================
// Banded PCR Finalize: X[i] = C[i] (after all dependencies eliminated)
// ============================================================================
__global__ void bpcr_finalize(
    unsigned int numRows,
    unsigned int numCols,
    float* C_coeff,
    float* xValues
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (int)numRows) return;
    for (unsigned int col = 0; col < numCols; col++) {
        xValues[row * numCols + col] = C_coeff[row * numCols + col];
    }
}

// ============================================================================
// Fallback kernel (kernel 2 logic) for matrices not suitable for BPCR
// ============================================================================
__global__ void kernel3_fallback(
    unsigned int  numRows,
    unsigned int* rowPtrs, unsigned int* colIdxs, float* matValues,
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
            if (row > 0) {
                while (atomicAdd(&dep_counter[row], 0) < (int)row_dep_count[row]) {
                    __nanosleep(10);
                }
            }
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

// ============================================================================
// Host: Detect matrix bandwidth, dispatch to BPCR or fallback
// ============================================================================
void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    // Detect maximum bandwidth (max distance from diagonal in subdiagonal entries)
    unsigned int max_bw = 0;
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            unsigned int c = L_r_host->colIdxs[j];
            if (c < i) {
                unsigned int dist = i - c;
                if (dist > max_bw) max_bw = dist;
            }
        }
    }

    CSRMatrix csr_shadow;
    cudaMemcpy(&csr_shadow, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix b_shadow, x_shadow;
    cudaMemcpy(&b_shadow, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x_shadow, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    // Use Banded PCR if bandwidth fits in BPCR_BLOCK
    if (max_bw < BPCR_BLOCK) {
        printf("  [BPCR mode: bandwidth=%u, using block size=%d]\n", max_bw, BPCR_BLOCK);

        unsigned int N_blocks = (n + BPCR_BLOCK - 1) / BPCR_BLOCK;
        size_t A_size = N_blocks * BPCR_BLOCK * BPCR_BLOCK * sizeof(float);
        size_t C_size = N_blocks * BPCR_BLOCK * numCols * sizeof(float);

        float *A_A, *A_B, *C_A, *C_B;
        cudaMalloc(&A_A, A_size); cudaMalloc(&A_B, A_size);
        cudaMalloc(&C_A, C_size); cudaMalloc(&C_B, C_size);

        // Init: build A[I] and C[I]
        dim3 init_grid(N_blocks);
        dim3 init_block((numCols > 64) ? numCols : 64);
        bpcr_init<<<init_grid, init_block>>>(
            n, csr_shadow.rowPtrs, csr_shadow.colIdxs, csr_shadow.values,
            b_shadow.values, numCols, A_A, C_A
        );

        // Apply PCR steps
        float *A_in = A_A, *C_in = C_A, *A_out = A_B, *C_out = C_B;
        int offset = 1;
        while (offset < (int)N_blocks) {
            bpcr_step<<<init_grid, init_block>>>(N_blocks, offset, A_in, C_in, A_out, C_out, numCols);
            float *t1 = A_in; A_in = A_out; A_out = t1;
            float *t2 = C_in; C_in = C_out; C_out = t2;
            offset <<= 1;
        }

        // Finalize: X[i] = C[i]
        dim3 fin_grid((n + 256 - 1) / 256);
        dim3 fin_block(256);
        bpcr_finalize<<<fin_grid, fin_block>>>(n, numCols, C_in, x_shadow.values);

        cudaFree(A_A); cudaFree(A_B); cudaFree(C_A); cudaFree(C_B);
    } else {
        printf("  [Fallback mode: bandwidth=%u too large for BPCR]\n", max_bw);

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
