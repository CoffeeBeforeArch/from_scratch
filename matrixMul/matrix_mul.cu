// This program computes matrix multiplication on the GPU using CUDA
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ void matrixMul(int *a, int *b, int *c, int N){
    // Calculate the global row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Matrix boundary check
    if(row < N && col < N){
        // Accumulate a partial result in tmp
        int tmp = 0;
        for(int i = 0; i < N; i++){
            tmp += a[row * N + i] * b[i * N + col];
        }
        
        // Write back the result
        c[row * N + col] = tmp;
    }

}

// Initialize a matrix with random numbers between 0 and 10
void init_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
}

// Verify the solution on the CPU
void verify_result(int *a, int *b, int *c, int N){
    // Temporary variable for partial results
    int tmp;

    // For every row...
    for(int i = 0; i < N; i++){
        // For every col...
        for(int j = 0; j < N; j++){
            // Multiply every element in the row-col pair
            tmp = 0;
            for(int k = 0; k < N; k++){
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check the result
            assert(tmp == c[i * N + j]);
        }
    }
}

int main(){
    // Dimensions of our square matrix (2^10 x 2^10 default)
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // Allocate memory
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set up kernel dimensions
    // Number of threads in x and y dimensions
    int THREADS = 16;
    // Number of blocks in x and y dimensions
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Setup the DIM3 kernel launch arguments
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch the kernel
    matrixMul<<<blocks, threads>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Call our functional test
    verify_result(a, b, c, N);

    cout << "PROGRAM COMPLETED SUCCESSFULLY!" << endl;

    return 0;
}
