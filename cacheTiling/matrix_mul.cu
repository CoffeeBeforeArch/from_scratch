// This program computes matrix multiplication on the GPU using shared
// memory
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

#define SHMEM_SIZE (16 * 16)

__global__ void matrixMul(int *a, int *b, int *c, int N){
    // Allocate shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Calculate each thread's global row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Extract some builtin values to simplify code
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    // Move the tile across the length of the grid
    int tmp = 0;
    for(int i = 0; i < (N / dim); i++){
        A[ty * dim + tx] = a[(row * N) + (i * dim) + tx];
        B[ty * dim + tx] = b[(i * dim * N) + (ty * N) + col];
        __syncthreads();

        // Accumulate the partial results
        for(int j = 0; j < dim; j++){
            tmp += A[ty * dim + j] * B[j * dim + tx];
        }
        __syncthreads();
    }

    // Write back the result to main memory
    c[row * N + col] = tmp;
}

// Initializes a square matrix with random numbers
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
    // Set the matrix dimensions (default = 2^10)
    int N = 1 << 12;
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for our matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Init our input matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set our CTA and Grid sizes
    int threads = 16;
    int blocks = (N + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Launch our kernel
    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Verify the result on the CPU
    //verify_result(a, b, c, N);

    cout << "PROGRAM COMPLETED SUCCESSFULLY!" << endl;

    return 0;
}
