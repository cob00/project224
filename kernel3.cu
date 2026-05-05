#include <stdio.h>
#include "common.h"
#include "matrix.h"


extern void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B,
                        DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host,
                        unsigned int numCols);


__global__ void pcr_init_kernel(
    float* __restrict__       d_alpha,
    float* __restrict__       d_beta,
    const float* __restrict__ bValues,
    const float* __restrict__ d_alpha1d,
    const float* __restrict__ d_inv_diag,
    int n, int numCols
)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= n || col >= numCols) return;
    int idx = row * numCols + col;
    d_alpha[idx] = d_alpha1d[row];
    d_beta[idx]  = bValues[idx] * d_inv_diag[row];
}

__global__ void pcr_kernel(
    float* d_alpha, float* d_beta,
    int n, int numCols, int stride
)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= n || col >= numCols) return;
    int idx = row * numCols + col;
    int src_row = row - stride;
    if (src_row >= 0) {
        int src_idx = src_row * numCols + col;
        float a_i   = d_alpha[idx];
        float a_src = d_alpha[src_idx];
        float b_src = d_beta[src_idx];
        d_beta[idx]  = a_i * b_src + d_beta[idx];
        d_alpha[idx] = a_i * a_src;
    }
}

__global__ void jacobi_kernel(
    const unsigned int* __restrict__ rowPtrs,
    const unsigned int* __restrict__ colIdxs,
    const float*        __restrict__ csrValues,
    const float*        __restrict__ bValues,
    const float*        __restrict__ x_old,
    float*              x_new,
    int n, int numCols
)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= n || col >= numCols) return;

    unsigned int rStart = rowPtrs[row];
    unsigned int rEnd   = rowPtrs[row + 1];

    float sum  = bValues[row * numCols + col];
    float diag = 1.0f;

    for (unsigned int j = rStart; j < rEnd; ++j) {
        unsigned int c = colIdxs[j];
        if (c < (unsigned int)row) {
            sum -= csrValues[j] * x_old[c * numCols + col];
        } else if (c == (unsigned int)row) {
            diag = csrValues[j];
        }
    }

    x_new[row * numCols + col] = (diag != 0.0f) ? (sum / diag) : 0.0f;
}

void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    // Analyse structure
    unsigned int sequential_count = 0;
    unsigned int max_dep_count = 0;
    for (unsigned int i = 1; i < n; ++i) {
        unsigned int dep_count = 0;
        bool depends_on_prev = false;
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            if (L_r_host->colIdxs[j] < i) {
                dep_count++;
                if (L_r_host->colIdxs[j] == i - 1) depends_on_prev = true;
            }
        }
        if (depends_on_prev) sequential_count++;
        if (dep_count > max_dep_count) max_dep_count = dep_count;
    }
    float seq_ratio = (n > 1) ? (float)sequential_count / (float)(n - 1) : 0.0f;

    bool use_pcr_jacobi = (seq_ratio >= 0.5f && max_dep_count <= 10);

    if (!use_pcr_jacobi) {
        sptrsv_gpu2(L_c, L_r, B, X, L_c_host, L_r_host, numCols);
        return;
    }

    float* alpha_h    = (float*)calloc(n, sizeof(float));
    float* inv_diag_h = (float*)calloc(n, sizeof(float));

    for (unsigned int i = 0; i < n; ++i) {
        float diag = 1.0f, sub = 0.0f;
        for (unsigned int j = L_r_host->rowPtrs[i]; j < L_r_host->rowPtrs[i + 1]; ++j) {
            unsigned int c = L_r_host->colIdxs[j];
            if (c == i) diag = L_r_host->values[j];
            else if (i > 0 && c == i - 1) sub = L_r_host->values[j];
        }
        alpha_h[i]    = (diag != 0.0f) ? (-sub / diag) : 0.0f;
        inv_diag_h[i] = (diag != 0.0f) ? (1.0f / diag) : 0.0f;
    }

    CSRMatrix csr_shadow;
    cudaMemcpy(&csr_shadow, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix b_shadow, x_shadow;
    cudaMemcpy(&b_shadow, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x_shadow, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    float* d_alpha1d;
    float* d_inv_diag;
    cudaMalloc(&d_alpha1d,  n * sizeof(float));
    cudaMalloc(&d_inv_diag, n * sizeof(float));
    cudaMemcpy(d_alpha1d,  alpha_h,    n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_diag, inv_diag_h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(alpha_h);
    free(inv_diag_h);

    size_t arr_size = (size_t)n * numCols * sizeof(float);
    float* d_alpha_arr;
    float* d_beta_arr;
    cudaMalloc(&d_alpha_arr, arr_size);
    cudaMalloc(&d_beta_arr,  arr_size);

    pcr_init_kernel<<<n, numCols>>>(
        d_alpha_arr, d_beta_arr,
        b_shadow.values, d_alpha1d, d_inv_diag,
        n, numCols
    );
    cudaFree(d_alpha1d);
    cudaFree(d_inv_diag);

    for (int stride = 1; stride < (int)n; stride *= 2) {
        pcr_kernel<<<n, numCols>>>(d_alpha_arr, d_beta_arr, n, numCols, stride);
    }

    cudaMemcpy(x_shadow.values, d_beta_arr, arr_size, cudaMemcpyDeviceToDevice);

    cudaFree(d_alpha_arr);
    cudaFree(d_beta_arr);

    float* d_x_temp;
    cudaMalloc(&d_x_temp, arr_size);

    float* x_old = x_shadow.values;  
    float* x_new = d_x_temp;

    int max_iters = 18;  
    for (int iter = 0; iter < max_iters; ++iter) {
        jacobi_kernel<<<n, numCols>>>(
            csr_shadow.rowPtrs,
            csr_shadow.colIdxs,
            csr_shadow.values,
            b_shadow.values,
            x_old,
            x_new,
            n, numCols
        );

        float* tmp = x_old;
        x_old = x_new;
        x_new = tmp;
    }

    if (x_old != x_shadow.values) {
        cudaMemcpy(x_shadow.values, x_old, arr_size, cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_x_temp);
}
