#include "common.h"
#include "matrix.h"

#define TILE 512
// Number of consecutive rows assigned to a single thread block.
// Larger = fewer blocks, less global-atomic contention, but more sequential
// work per block. 8 is a reasonable starting point for chain-heavy matrices
// like tmt_sym; tune by trying 4, 8, 16, 32.
#define ROWS_PER_BLOCK 8

// Second optimization: multi-row blocks.
//
// In kernel1, every row is its own block, and every block spins on
// dep_counter[row] in global memory until its parents finish. For matrices
// with chain-like dependencies (e.g. tmt_sym, where row i depends on row i-1),
// this means almost every block spends most of its life spinning on a global
// atomic — which serializes through the L2 atomic units.
//
// This kernel groups ROWS_PER_BLOCK consecutive rows into a single block and
// processes them sequentially within the block. Dependencies between rows in
// the SAME block are resolved by __syncthreads() (on-chip, fast). Only
// dependencies on rows in EARLIER blocks require a global spin-wait, which
// happens at most once per row at the start (and trivially completes when
// cross_block_dep_count == 0). Similarly, child signaling via atomicAdd
// only happens for children in LATER blocks; intra-block children are
// fed via cached writes + syncthreads.
//
// Also uses a volatile read instead of atomicAdd(addr, 0) for the spin-wait,
// avoiding the L2 atomic unit on the read side.
__global__ void sptrsv_kernel2(
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
    unsigned int* cross_block_dep_count
) {
    unsigned int block_start    = blockIdx.x * ROWS_PER_BLOCK;
    unsigned int block_end_excl = block_start + ROWS_PER_BLOCK;
    if (block_end_excl > numRows) block_end_excl = numRows;

    unsigned int col = threadIdx.x;
    if (col >= numCols) return;

    extern __shared__ char smem[];
    unsigned int* s_cols = (unsigned int*) smem;
    float*        s_vals = (float*) (smem + TILE * sizeof(unsigned int));

    volatile int* dep_v = (volatile int*) dep_counter;

    // Process each row in this block sequentially.
    for (unsigned int row = block_start; row < block_end_excl; ++row) {

        // 1. Wait for cross-block dependencies of this row.
        //    Intra-block parents (in [block_start, row)) are already done
        //    because we processed them in earlier iterations.
        unsigned int needed = cross_block_dep_count[row];
        if (col == 0 && needed > 0) {
            while (dep_v[row] < (int) needed) {
                __nanosleep(10);
            }
        }
        __syncthreads();
        __threadfence();

        // 2. Forward substitution for this row.
        unsigned int rowStart = rowPtrs[row];
        unsigned int rowEnd   = rowPtrs[row + 1];

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

            for (unsigned int j = 0; j < tileSize; ++j) {
                unsigned int c   = s_cols[j];
                float        val = s_vals[j];
                if (c < row) {
                    sum -= val * xValues[c * numCols + col];
                } else if (c == row) {
                    diag = val != 0.0f ? val : 1.0f;
                }
            }
            __syncthreads();
        }

        xValues[row * numCols + col] = sum / diag;
        // Make this row's writes visible to the next iteration (intra-block
        // children) and to other blocks (cross-block children).
        __syncthreads();
        __threadfence();

        // 3. Signal cross-block children.
        //    Intra-block children (with index in [row+1, block_end_excl)) are
        //    going to be processed in the next iterations of this very loop —
        //    they read xValues directly, no signaling needed.
        if (col == 0) {
            unsigned int cStart = cscColPtrs[row];
            unsigned int cEnd   = cscColPtrs[row + 1];
            for (unsigned int j = cStart; j < cEnd; ++j) {
                unsigned int dep = cscRowIdxs[j];
                if (dep >= block_end_excl) {
                    atomicAdd(&dep_counter[dep], 1);
                }
            }
        }
    }
}


void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    // Precompute cross_block_dep_count on the host:
    // for each row, count parents whose row index is in an EARLIER block.
    // (Parents in the same block are handled by intra-block syncthreads.)
    unsigned int* cross_dep_h = (unsigned int*) calloc(n, sizeof(unsigned int));
    for (unsigned int row = 0; row < n; ++row) {
        unsigned int block_start = (row / ROWS_PER_BLOCK) * ROWS_PER_BLOCK;
        unsigned int count = 0;
        for (unsigned int j = L_r_host->rowPtrs[row]; j < L_r_host->rowPtrs[row + 1]; ++j) {
            unsigned int c = L_r_host->colIdxs[j];
            if (c < block_start) count++;
        }
        cross_dep_h[row] = count;
    }

    unsigned int* cross_dep_d;
    int*          dep_counter_d;
    cudaMalloc(&cross_dep_d,    n * sizeof(unsigned int));
    cudaMalloc(&dep_counter_d,  n * sizeof(int));
    cudaMemcpy(cross_dep_d, cross_dep_h, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(dep_counter_d, 0, n * sizeof(int));
    free(cross_dep_h);

    // Pull device pointers out of the device-side struct wrappers
    CSRMatrix csr_shadow;
    cudaMemcpy(&csr_shadow, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    CSCMatrix csc_shadow;
    cudaMemcpy(&csc_shadow, L_c, sizeof(CSCMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix b_shadow, x_shadow;
    cudaMemcpy(&b_shadow, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x_shadow, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    unsigned int smemSize = TILE * (sizeof(unsigned int) + sizeof(float));

    unsigned int numBlocks = (n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(numBlocks);
    dim3 block(numCols);

    sptrsv_kernel2<<<grid, block, smemSize>>>(
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
        cross_dep_d
    );

    cudaFree(dep_counter_d);
    cudaFree(cross_dep_d);
}
