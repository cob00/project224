
#include "common.h"
#include "matrix.h"

#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <random>

#include <cstdio>
#include <cstdlib>
#include <algorithm>

struct Entry {
    unsigned int src;
    unsigned int dst;
    double val;
};

CSCMatrix* createCSCMatrixFromFile(const char* fileName){
    if (fileName == nullptr) {
        std::fprintf(stderr, "Error: fileName is null.\n");
        return nullptr;
    }

    FILE* file = std::fopen(fileName, "r");
    if (!file) {
        std::fprintf(stderr, "Error: could not open file %s\n", fileName);
        return nullptr;
    }

    unsigned int numRows = 0;
    unsigned int numNonzeros = 0;

    if (std::fscanf(file, "%u %u", &numRows, &numNonzeros) != 2) {
        std::fprintf(stderr, "Error: failed to read header from file %s\n", fileName);
        std::fclose(file);
        return nullptr;
    }

    unsigned int maxPossibleNonzeros = numNonzeros + numRows;

    Entry* entries = (Entry*) std::malloc(sizeof(Entry) * maxPossibleNonzeros);
    unsigned int* diagPresent = (unsigned int*) std::calloc(numRows, sizeof(unsigned int));

    if (!entries || !diagPresent) {
        std::fprintf(stderr, "Error: memory allocation failed.\n");
        std::fclose(file);
        std::free(entries);
        std::free(diagPresent);
        return nullptr;
    }

    unsigned int actualNonzeros = 0;

    while (actualNonzeros < numNonzeros) {
        unsigned int src, dst;
        double val;

        int ret = std::fscanf(file, "%u %u %lf", &src, &dst, &val);
        if (ret == EOF) {
            break;
        }
        if (ret != 3) {
            std::fprintf(stderr, "Error: malformed line at entry %u\n", actualNonzeros);
            std::free(entries);
            std::free(diagPresent);
            std::fclose(file);
            return nullptr;
        }

        if (src >= numRows || dst >= numRows) {
            std::fprintf(stderr,
                         "Error: index out of bounds at entry %u: src=%u dst=%u\n",
                         actualNonzeros, src, dst);
            std::free(entries);
            std::free(diagPresent);
            std::fclose(file);
            return nullptr;
        }

        entries[actualNonzeros].src = src;
        entries[actualNonzeros].dst = dst;
        entries[actualNonzeros].val = val;

        if (src == dst) {
            diagPresent[src] = 1;
        }

        actualNonzeros++;
    }

    std::fclose(file);

    if (actualNonzeros != numNonzeros) {
        std::fprintf(stderr,
                     "Warning: header says %u nonzeros, but file contains %u entries.\n",
                     numNonzeros, actualNonzeros);
    }

    // Add missing diagonal entries with value 1.0
    for (unsigned int i = 0; i < numRows; i++) {
        if (!diagPresent[i]) {
            entries[actualNonzeros].src = i;
            entries[actualNonzeros].dst = i;
            entries[actualNonzeros].val = 1.0;
            actualNonzeros++;
        }
    }

    std::free(diagPresent);

    // Sort by column first, then by row
    std::sort(entries, entries + actualNonzeros,
        [](const Entry& a, const Entry& b) {
            if (a.dst != b.dst) return a.dst < b.dst;
            return a.src < b.src;
        }
    );

    CSCMatrix* mat = (CSCMatrix*) std::malloc(sizeof(CSCMatrix));
    if (!mat) {
        std::fprintf(stderr, "Error: memory allocation failed for CSCMatrix.\n");
        std::free(entries);
        return nullptr;
    }

    mat->numRows = numRows;
    mat->numCols = numRows;
    mat->numNonzeros = actualNonzeros;
    mat->colPtrs = (unsigned int*) std::calloc(numRows + 1, sizeof(unsigned int));
    mat->rowIdxs = (unsigned int*) std::malloc(sizeof(unsigned int) * actualNonzeros);
    mat->colIdxs = (unsigned int*) std::malloc(sizeof(unsigned int) * actualNonzeros);
    mat->values  = (float*) std::malloc(sizeof(float) * actualNonzeros);

    if (!mat->colPtrs || !mat->rowIdxs || !mat->values) {
        std::fprintf(stderr, "Error: memory allocation failed for CSC arrays.\n");
        std::free(entries);
        std::free(mat->colPtrs);
        std::free(mat->rowIdxs);
        std::free(mat->colIdxs);
        std::free(mat->values);
        std::free(mat);
        return nullptr;
    }

    // Count entries per column
    for (unsigned int i = 0; i < actualNonzeros; i++) {
        mat->colPtrs[entries[i].dst + 1]++;
    }

    // Prefix sum
    for (unsigned int col = 0; col < numRows; col++) {
        mat->colPtrs[col + 1] += mat->colPtrs[col];
    }

    // Since entries are already sorted by column then row,
    // we can directly copy them in order
    for (unsigned int i = 0; i < actualNonzeros; i++) {
        mat->rowIdxs[i] = entries[i].src;
        mat->colIdxs[i] = entries[i].dst;
        mat->values[i] = (float) entries[i].val;
    }

    std::free(entries);
    return mat;
}

DenseMatrix* generateDenseMatrix(unsigned int numRows, unsigned int numCols){
    DenseMatrix* mat = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    mat->numRows = numRows;
    mat->numCols = numCols;
    mat->values = (float*)malloc(sizeof(float) * numRows * numCols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (unsigned int i = 0; i < numRows; i++) {
        for (unsigned int j = 0; j < numCols; j++) {
            mat->values[i * numCols + j] = dist(gen);
        }
    }

    return mat;
}


CSRMatrix* createCSRMatrixFromCSCMatrix(CSCMatrix* cscMatrix){
    unsigned int n = cscMatrix->numRows;
    CSRMatrix* csrMatrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    csrMatrix->numRows = n;
    csrMatrix->numCols = n;
    csrMatrix->numNonzeros = cscMatrix->numNonzeros;
    csrMatrix->rowPtrs = (unsigned int*)malloc(sizeof(unsigned int) * (n + 1));
    csrMatrix->colIdxs = (unsigned int*)malloc(sizeof(unsigned int) * cscMatrix->numNonzeros);
    csrMatrix->rowIdxs = (unsigned int*)malloc(sizeof(unsigned int) * cscMatrix->numNonzeros);
    csrMatrix->values = (float*)malloc(sizeof(float) * cscMatrix->numNonzeros);
    memset(csrMatrix->rowPtrs, 0, sizeof(unsigned int) * (n + 1));

    if (csrMatrix->rowPtrs == nullptr || csrMatrix->colIdxs == nullptr || csrMatrix->values == nullptr) {
        free(csrMatrix->rowPtrs);
        free(csrMatrix->colIdxs);
        free(csrMatrix->values);
        free(csrMatrix);
        printf("Error: failed to allocate memory for CSR matrix.\n");
        return nullptr;
    }
    for (unsigned int col = 0; col < cscMatrix->numCols; col++) {
        for (unsigned int idx = cscMatrix->colPtrs[col]; idx < cscMatrix->colPtrs[col + 1]; idx++) {
            unsigned int row = cscMatrix->rowIdxs[idx];
            csrMatrix->rowPtrs[row + 1]++;
        }
    }

    for (unsigned int i = 0; i < n; i++) {
        csrMatrix->rowPtrs[i + 1] += csrMatrix->rowPtrs[i];
    }

    unsigned int* nextPos = (unsigned int*)malloc(sizeof(unsigned int) * n);
    if (nextPos == nullptr) {
        free(csrMatrix->rowPtrs);
        free(csrMatrix->colIdxs);
        free(csrMatrix->values);
        free(csrMatrix);
        return nullptr;
    }

    for (unsigned int i = 0; i < n; i++) {
        nextPos[i] = csrMatrix->rowPtrs[i];
    }

    for (unsigned int col = 0; col < cscMatrix->numCols; col++) {
        for (unsigned int idx = cscMatrix->colPtrs[col]; idx < cscMatrix->colPtrs[col + 1]; idx++) {
            unsigned int row = cscMatrix->rowIdxs[idx];
            unsigned int pos = nextPos[row]++;

            csrMatrix->colIdxs[pos] = col;
            csrMatrix->rowIdxs[pos] = row;
            csrMatrix->values[pos] = cscMatrix->values[idx];
        }
    }
    free(nextPos);
    return csrMatrix;
}

void freeCSCMatrix(CSCMatrix* mat) {
    if (mat) {
        free(mat->colPtrs);
        free(mat->rowIdxs);
        free(mat->colIdxs);
        free(mat->values);
        free(mat);
    }
}

void freeCSRMatrix(CSRMatrix* mat) {
    if (mat) {
        free(mat->rowPtrs);
        free(mat->colIdxs);
        free(mat->rowIdxs);
        free(mat->values);
        free(mat);
    }
}

void freeDenseMatrix(DenseMatrix* mat) {
    if (mat) {
        free(mat->values);
        free(mat);
    }
}

void copyCSCMatrixToGPU(CSCMatrix* cscMatrix_h, CSCMatrix* cscMatrix_d) {
    CSCMatrix cscMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&cscMatrixShadow, cscMatrix_d, sizeof(CSCMatrix), cudaMemcpyDeviceToHost));
    assert(cscMatrixShadow.numRows == cscMatrix_h->numRows);
    assert(cscMatrixShadow.numCols == cscMatrix_h->numCols);
    assert(cscMatrixShadow.numNonzeros == cscMatrix_h->numNonzeros);
    CUDA_ERROR_CHECK(cudaMemcpy(cscMatrixShadow.colPtrs, cscMatrix_h->colPtrs, (cscMatrix_h->numCols + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(cscMatrixShadow.rowIdxs, cscMatrix_h->rowIdxs, cscMatrix_h->numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(cscMatrixShadow.colIdxs, cscMatrix_h->colIdxs, cscMatrix_h->numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(cscMatrixShadow.values, cscMatrix_h->values, cscMatrix_h->numNonzeros*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void copyCSRMatrixToGPU(CSRMatrix* csrMatrix_h, CSRMatrix* csrMatrix_d) {
    CSRMatrix csrMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&csrMatrixShadow, csrMatrix_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost));
    assert(csrMatrixShadow.numRows == csrMatrix_h->numRows);
    assert(csrMatrixShadow.numCols == csrMatrix_h->numCols);
    assert(csrMatrixShadow.numNonzeros == csrMatrix_h->numNonzeros);
    CUDA_ERROR_CHECK(cudaMemcpy(csrMatrixShadow.rowPtrs, csrMatrix_h->rowPtrs, (csrMatrix_h->numRows + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(csrMatrixShadow.colIdxs, csrMatrix_h->colIdxs, csrMatrix_h->numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(csrMatrixShadow.rowIdxs, csrMatrix_h->rowIdxs, csrMatrix_h->numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(csrMatrixShadow.values, csrMatrix_h->values, csrMatrix_h->numNonzeros*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void copyDenseMatrixToGPU(DenseMatrix* denseMatrix_h, DenseMatrix* denseMatrix_d) {
    DenseMatrix denseMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&denseMatrixShadow, denseMatrix_d, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    denseMatrixShadow.numRows = denseMatrix_h->numRows;
    denseMatrixShadow.numCols = denseMatrix_h->numCols;
    assert(denseMatrixShadow.numRows == denseMatrix_h->numRows);
    assert(denseMatrixShadow.numCols == denseMatrix_h->numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(denseMatrixShadow.values, denseMatrix_h->values, (denseMatrix_h->numRows * denseMatrix_h->numCols)*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void copyDenseMatrixFromGPU(DenseMatrix* denseMatrix_d, DenseMatrix* denseMatrix_h) {
    DenseMatrix denseMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&denseMatrixShadow, denseMatrix_d, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    assert(denseMatrixShadow.numRows == denseMatrix_h->numRows);
    assert(denseMatrixShadow.numCols == denseMatrix_h->numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(denseMatrix_h->values, denseMatrixShadow.values, (denseMatrix_h->numRows * denseMatrix_h->numCols)*sizeof(float), cudaMemcpyDeviceToHost));
}

DenseMatrix* createEmptyDenseMatrix(unsigned int numRows, unsigned int numCols){
    DenseMatrix* mat = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    mat->numRows = numRows;
    mat->numCols = numCols;
    mat->values = (float*)malloc(sizeof(float) * numRows * numCols);
    return mat;
}

CSCMatrix* createEmptyCSCMatrixOnGPU(unsigned int numCols, unsigned int numNonzeros) {

    CSCMatrix cscMatrixShadow;
    cscMatrixShadow.numCols = numCols;
    cscMatrixShadow.numRows = numCols; // square matrix
    cscMatrixShadow.numNonzeros = numNonzeros;
    CUDA_ERROR_CHECK(cudaMalloc((void**) &cscMatrixShadow.colPtrs, (numCols+1)*sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &cscMatrixShadow.rowIdxs, numNonzeros*sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &cscMatrixShadow.colIdxs, numNonzeros*sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &cscMatrixShadow.values, numNonzeros*sizeof(float)));

    CSCMatrix* cscMatrix;
    CUDA_ERROR_CHECK(cudaMalloc((void**) &cscMatrix, sizeof(CSCMatrix)));
    CUDA_ERROR_CHECK(cudaMemcpy(cscMatrix, &cscMatrixShadow, sizeof(CSCMatrix), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return cscMatrix;

}

DenseMatrix* createEmptyDenseMatrixOnGPU(unsigned int numRows, unsigned int numCols) {

    DenseMatrix denseMatrixShadow;
    denseMatrixShadow.numRows = numRows;
    denseMatrixShadow.numCols = numCols;
    CUDA_ERROR_CHECK(cudaMalloc((void**) &denseMatrixShadow.values, numRows*numCols*sizeof(float)));

    DenseMatrix* denseMatrix;
    CUDA_ERROR_CHECK(cudaMalloc((void**) &denseMatrix, sizeof(DenseMatrix)));
    CUDA_ERROR_CHECK(cudaMemcpy(denseMatrix, &denseMatrixShadow, sizeof(DenseMatrix), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return denseMatrix;

}
CSRMatrix* createEmptyCSRMatrixOnGPU(unsigned int numRows, unsigned int numNonzeros) {

    CSRMatrix csrMatrixShadow;
    csrMatrixShadow.numRows = numRows;
    csrMatrixShadow.numCols = numRows; // square matrix
    csrMatrixShadow.numNonzeros = numNonzeros;
    CUDA_ERROR_CHECK(cudaMalloc((void**) &csrMatrixShadow.rowPtrs, (numRows+1)*sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &csrMatrixShadow.colIdxs, numNonzeros*sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &csrMatrixShadow.rowIdxs, numNonzeros*sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc((void**) &csrMatrixShadow.values, numNonzeros*sizeof(float)));

    CSRMatrix* csrMatrix;
    CUDA_ERROR_CHECK(cudaMalloc((void**) &csrMatrix, sizeof(CSRMatrix)));
    CUDA_ERROR_CHECK(cudaMemcpy(csrMatrix, &csrMatrixShadow, sizeof(CSRMatrix), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return csrMatrix;

}

void freeCSCMatrixOnGPU(CSCMatrix* cscMatrix) {
    CSCMatrix cscMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&cscMatrixShadow, cscMatrix, sizeof(CSCMatrix), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(cscMatrixShadow.colPtrs));
    CUDA_ERROR_CHECK(cudaFree(cscMatrixShadow.rowIdxs));
    CUDA_ERROR_CHECK(cudaFree(cscMatrixShadow.colIdxs));
    CUDA_ERROR_CHECK(cudaFree(cscMatrixShadow.values));
    CUDA_ERROR_CHECK(cudaFree(cscMatrix));
}

void freeCSRMatrixOnGPU(CSRMatrix* csrMatrix) {
    CSRMatrix csrMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&csrMatrixShadow, csrMatrix, sizeof(CSRMatrix), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(csrMatrixShadow.rowPtrs));
    CUDA_ERROR_CHECK(cudaFree(csrMatrixShadow.colIdxs));
    CUDA_ERROR_CHECK(cudaFree(csrMatrixShadow.rowIdxs));
    CUDA_ERROR_CHECK(cudaFree(csrMatrixShadow.values));
    CUDA_ERROR_CHECK(cudaFree(csrMatrix));
}

void freeDenseMatrixOnGPU(DenseMatrix* denseMatrix) {
    DenseMatrix denseMatrixShadow;
    CUDA_ERROR_CHECK(cudaMemcpy(&denseMatrixShadow, denseMatrix, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(denseMatrixShadow.values));
    CUDA_ERROR_CHECK(cudaFree(denseMatrix));
}