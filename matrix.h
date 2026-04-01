#ifndef __MATRIX_H_
#define __MATRIX_H_

struct CSCMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* colPtrs;
    unsigned int* rowIdxs;
    unsigned int* colIdxs; 
    float* values;
};


CSCMatrix* createCSCMatrixFromFile(const char* fileName);
void copyCSCMatrixToGPU(CSCMatrix* cscMatrix_h, CSCMatrix* cscMatrix_d);
CSCMatrix* createEmptyCSCMatrixOnGPU(unsigned int numCols, unsigned int numNonzeros); 
void freeCSCMatrix(CSCMatrix* cscMatrix);
void freeCSCMatrixOnGPU(CSCMatrix* cscMatrix);

struct DenseMatrix {
    unsigned int numRows;
    unsigned int numCols;
    float* values;
};

DenseMatrix* generateDenseMatrix(unsigned int numRows, unsigned int numCols);
void copyDenseMatrixToGPU(DenseMatrix* denseMatrix_h, DenseMatrix* denseMatrix_d);
void copyDenseMatrixFromGPU(DenseMatrix* denseMatrix_d, DenseMatrix* denseMatrix_h);
DenseMatrix* createEmptyDenseMatrix(unsigned int numRows, unsigned int numCols);
DenseMatrix* createEmptyDenseMatrixOnGPU(unsigned int numRows, unsigned int numCols);
void freeDenseMatrixOnGPU(DenseMatrix* denseMatrix);
void freeDenseMatrix(DenseMatrix* denseMatrix);

struct CSRMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    unsigned int* rowIdxs;
    float* values;
};

CSRMatrix* createCSRMatrixFromCSCMatrix(CSCMatrix* cscMatrix);
void copyCSRMatrixToGPU(CSRMatrix* csrMatrix_h, CSRMatrix* csrMatrix_d);
CSRMatrix* createEmptyCSRMatrixOnGPU(unsigned int numRows, unsigned int numNonzeros);
void freeCSRMatrix(CSRMatrix* csrMatrix);
void freeCSRMatrixOnGPU(CSRMatrix* csrMatrix);

#endif