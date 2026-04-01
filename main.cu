#include <assert.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "common.h"
#include "matrix.h"
#include "timer.h"

void verify(DenseMatrix* result, DenseMatrix* reference) {
    printf("    Verifying result\n");
    for(unsigned int i = 0; i < reference->numRows; ++i) {
        for(unsigned int j = 0; j < reference->numCols; ++j){
            unsigned int idx = i * reference->numCols + j;
            float refVal = reference->values[idx];
            float resVal = result->values[idx];
            if(fabs(resVal - refVal)/refVal > 1e-4) {
                printf("        Mismatch at (%u, %u): computed=%f, reference=%f\n", i, j, result->values[idx], reference->values[idx]);
                return;
            }
        }
        
    }
    printf("        Verification succeeded\n");
}

void (*sptrsv_gpu[])(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols) = {
    sptrsv_gpu0,
    sptrsv_gpu1,
    sptrsv_gpu2,
    sptrsv_gpu3
};


int main(int argc, char* argv[]) {

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    setbuf(stdout, NULL);
    const char* dataset = "data/rajat18.txt";
    unsigned int runCPUVersion = 1;
    unsigned int runGPUVersion[4] = { 0 };
    unsigned int useGPU = 0;
    int opt;
    // Parse arguments
    while ((opt = getopt(argc, argv, "d:s0123")) >= 0) {
        switch (opt) {
            case 'd':
                if (strcmp(optarg, "s") == 0) {
                    dataset = "data/rajat18.txt";
                } else if (strcmp(optarg, "m") == 0) {
                    dataset = "data/parabolic_fem.txt";
                } else if (strcmp(optarg, "l") == 0) {
                    dataset = "data/tmt_sym.txt";
                } else {
                    fprintf(stderr, "Invalid dataset size. Use -d s, -d m, or -d l\n");
                    exit(1);
                }
                break;

            case 's':
                runCPUVersion = 1;
                break;

            case '0':
                runGPUVersion[0] = 1;
                useGPU = 1;
                break;
            case '1':
                runGPUVersion[1] = 1;
                useGPU = 1;
                break;
            case '2':
                runGPUVersion[2] = 1;
                useGPU = 1;
                break;
            case '3':
                runGPUVersion[3] = 1;
                useGPU = 1;
                break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                exit(1);
        }
    }
    // Allocate memory and intilalize data
    CSCMatrix* csc_h = createCSCMatrixFromFile(dataset);
    CSRMatrix* csr_h = createCSRMatrixFromCSCMatrix(csc_h);

    // I want to create 3 matrices rather than 1 I want them with 128,256,512 colums 
    DenseMatrix* dense_h_128 = generateDenseMatrix(csr_h->numRows, 128);
    DenseMatrix* dense_h_256 = generateDenseMatrix(csr_h->numRows, 256);
    DenseMatrix* dense_h_512 = generateDenseMatrix(csr_h->numRows, 512);
    DenseMatrix* result_h_128 = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    DenseMatrix* result_h_256 = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    DenseMatrix* result_h_512 = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    DenseMatrix* result_gpu_128 = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    DenseMatrix* result_gpu_256 = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    DenseMatrix* result_gpu_512 = (DenseMatrix*)malloc(sizeof(DenseMatrix));
    result_h_128->numCols = 128;
    result_h_256->numCols = 256;
    result_h_512->numCols = 512;
    result_gpu_128->numCols = 128;
    result_gpu_256->numCols = 256;
    result_gpu_512->numCols = 512;
    result_h_128->numRows = csr_h->numRows;
    result_h_256->numRows = csr_h->numRows;
    result_h_512->numRows = csr_h->numRows;
    result_gpu_128->numRows = csr_h->numRows;
    result_gpu_256->numRows = csr_h->numRows;
    result_gpu_512->numRows = csr_h->numRows;
    result_h_128->values = (float*)malloc(result_h_128->numRows * result_h_128->numCols * sizeof(float));
    result_h_256->values = (float*)malloc(result_h_256->numRows * result_h_256->numCols * sizeof(float));
    result_h_512->values = (float*)malloc(result_h_512->numRows * result_h_512->numCols * sizeof(float));
    result_gpu_128->values = (float*)malloc(result_gpu_128->numRows * result_gpu_128->numCols * sizeof(float));
    result_gpu_256->values = (float*)malloc(result_gpu_256->numRows * result_gpu_256->numCols * sizeof(float));
    result_gpu_512->values = (float*)malloc(result_gpu_512->numRows * result_gpu_512->numCols * sizeof(float));

    
    

    if(runCPUVersion) {
        printf("Running CPU version...\n");
        
        // Compute on CPU
        Timer timer;
        startTime(&timer);
        sptrsv_cpu(csr_h, dense_h_128, result_h_128);
        stopTime(&timer);
        printElapsedTime(timer, "   CPU time(128 cols)", CYAN);

        startTime(&timer);
        sptrsv_cpu(csr_h, dense_h_256, result_h_256);
        stopTime(&timer);
        printElapsedTime(timer, "   CPU time(256 cols)", CYAN);

        startTime(&timer);
        sptrsv_cpu(csr_h, dense_h_512, result_h_512);
        stopTime(&timer);
        printElapsedTime(timer, "   CPU time(512 cols)", CYAN);

    }

    if(useGPU){
        
        // Allocate GPU memory
        CSCMatrix* csc_d = createEmptyCSCMatrixOnGPU(csc_h->numCols, csc_h->numNonzeros);
        CSRMatrix* csr_d = createEmptyCSRMatrixOnGPU(csr_h->numRows, csr_h->numNonzeros);
        DenseMatrix* dense_d_128 = createEmptyDenseMatrixOnGPU(dense_h_128->numRows, dense_h_128->numCols);
        DenseMatrix* dense_d_256 = createEmptyDenseMatrixOnGPU(dense_h_256->numRows, dense_h_256->numCols);
        DenseMatrix* dense_d_512 = createEmptyDenseMatrixOnGPU(dense_h_512->numRows, dense_h_512->numCols);
        DenseMatrix* result_d_128 = createEmptyDenseMatrixOnGPU(result_h_128->numRows, result_h_128->numCols); 
        DenseMatrix* result_d_256 = createEmptyDenseMatrixOnGPU(result_h_256->numRows, result_h_256->numCols);
        DenseMatrix* result_d_512 = createEmptyDenseMatrixOnGPU(result_h_512->numRows, result_h_512->numCols);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        
        

        // Copy data to GPU
        copyCSCMatrixToGPU(csc_h, csc_d);
        copyCSRMatrixToGPU(csr_h, csr_d);
        copyDenseMatrixToGPU(dense_h_128, dense_d_128);
        copyDenseMatrixToGPU(dense_h_256, dense_d_256);
        copyDenseMatrixToGPU(dense_h_512, dense_d_512);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        
        for(unsigned int gpuVersion = 0; gpuVersion < 4; ++gpuVersion){
            if(runGPUVersion[gpuVersion]){
               
                printf("Running GPU version %u...\n", gpuVersion);
                
                // Compute on GPU
                Timer timer;
                startTime(&timer);
                sptrsv_gpu[gpuVersion](csc_d, csr_d, dense_d_128, result_d_128, csc_h, csr_h, dense_h_128->numCols);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer);
                printElapsedTime(timer, "    GPU kernel time(128 cols)", GREEN);

                // Copy data from GPU
                copyDenseMatrixFromGPU(result_d_128, result_gpu_128);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());

                // Verify
                verify(result_gpu_128, result_h_128);

                // Free GPU memory for this version
                freeDenseMatrixOnGPU(result_d_128);
                result_d_128 = createEmptyDenseMatrixOnGPU(result_h_128->numRows, result_h_128->numCols);
                copyCSRMatrixToGPU(csr_h, csr_d);
                copyCSCMatrixToGPU(csc_h, csc_d);


                // Compute on GPU
                startTime(&timer);
                sptrsv_gpu[gpuVersion](csc_d, csr_d, dense_d_256, result_d_256, csc_h, csr_h, dense_h_256->numCols);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer);
                printElapsedTime(timer, "    GPU kernel time(256 cols)", GREEN);    

                // Copy data from GPU
                copyDenseMatrixFromGPU(result_d_256, result_gpu_256);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());

                // Verify
                verify(result_gpu_256, result_h_256);

                // Free GPU memory for this version
                freeDenseMatrixOnGPU(result_d_256);
                result_d_256 = createEmptyDenseMatrixOnGPU(result_h_256->numRows, result_h_256->numCols);
                copyCSRMatrixToGPU(csr_h, csr_d);
                copyCSCMatrixToGPU(csc_h, csc_d);
                
                // Compute on GPU
                startTime(&timer);
                sptrsv_gpu[gpuVersion](csc_d, csr_d, dense_d_512, result_d_512, csc_h, csr_h, dense_h_512->numCols);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer);
                printElapsedTime(timer, "    GPU kernel time(512 cols)", GREEN);

                // Copy data from GPU
                copyDenseMatrixFromGPU(result_d_512, result_gpu_512);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());

                // Verify
                verify(result_gpu_512, result_h_512);

                // Free GPU memory for this version
                freeDenseMatrixOnGPU(result_d_512);
                result_d_512 = createEmptyDenseMatrixOnGPU(result_h_512->numRows, result_h_512->numCols);
                copyCSRMatrixToGPU(csr_h, csr_d);
                copyCSCMatrixToGPU(csc_h, csc_d);
        
            }
        }

        // Free GPU memory
        freeDenseMatrixOnGPU(dense_d_128);
        freeDenseMatrixOnGPU(result_d_128);
        freeDenseMatrixOnGPU(dense_d_256);
        freeDenseMatrixOnGPU(result_d_256);
        freeDenseMatrixOnGPU(dense_d_512);
        freeDenseMatrixOnGPU(result_d_512);
        freeCSRMatrixOnGPU(csr_d);
        freeCSCMatrixOnGPU(csc_d);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    // Free CPU memory
    freeDenseMatrix(dense_h_128);
    freeDenseMatrix(dense_h_256);
    freeDenseMatrix(dense_h_512);
    free(result_h_128->values);
    free(result_h_256->values);
    free(result_h_512->values);
    freeCSRMatrix(csr_h);
    freeCSCMatrix(csc_h);
    free(result_h_128);
    free(result_h_256);
    free(result_h_512);

    return 0;
}
    

