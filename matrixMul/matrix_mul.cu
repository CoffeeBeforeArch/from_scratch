// This program computes matrix multiplication on the GPU using CUDA
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ void matrixMul(int *a, int *b, int *c, int N){
    // Calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for our matrix
    if(row < N && col < N){
        // Accumulate a partial result
        int tmp = 0;
        for(int i = 0; i < N; i++){
            tmp += a[row * N + i] * b[i * N + col];
        }

        // Write back the result
        c[row * N + col] = tmp;
    }
}

// Initializes a square matrix with random numbers between 0-100
void init_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
}

// Verify the result on the CPU
void verify_result(int *a, int *b, int *c, int N){
    int tmp;
    // For every row...
    for(int i = 0; i < N; i++){
        // For every col...
        for(int j = 0; j < N; j++){
            // For every element in the row-col pair
            tmp = 0;
            for(int k = 0; k < N; k++){
                tmp += a[i * N + k] * b[k * N + j];
            }
            
            // Check each result
            assert(tmp == c[i * N + j]);
        }
    }
}

int main(){
    // Set our square matrix dimension (2^10 x 2^10 default) 
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for our matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set our CTA and Grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // Setup our kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Launch our kernel
    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    cout << "PROGRAM COMPLETED SUCCESSFULLY!" << endl;
    
    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
