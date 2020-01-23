// This program computes matrix multiplication on the GPU using CUDA
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>
#include <iostream>

#define N 1024

// Declare our own type which is an int array of known width
typedef int my_array[N];

__global__ void matrixMul(my_array *a, my_array *b, my_array *c){
    // Calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for our matrix
    if(row < N && col < N){
        // Accumulate a partial result
        int tmp = 0;
        for(int i = 0; i < N; i++){
            tmp += a[row][i] * b[i][col];
        }

        // Write back the result
        c[row][col] = tmp;
    }
}

// Initializes a square matrix with random numbers between 0-100
void init_matrix(my_array *m){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      m[i][j] = rand() % 100;
    }
  }
}

// Verify the result on the CPU
void verify_result(my_array *a, my_array *b, my_array *c){
  // For every row...
  for(int i = 0; i < N; i++){
    // For every col...
    for(int j = 0; j < N; j++){
      // For every element in the row-col pair
      int tmp = 0;
      for(int k = 0; k < N; k++){
        tmp += a[i][k] * b[k][j];
      }
            
      // Check each result
      assert(tmp == c[i][j]);
    }
  }
}

int main(){
    // Set our square matrix dimension (2^10 x 2^10 default) 
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for our matrices
    my_array *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize our matrices
    init_matrix(a);
    init_matrix(b);

    // Set our CTA and Grid dimensions
    int threads = 32;
    int blocks = (N + threads - 1) / threads;

    // Setup our kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Launch our kernel
    matrixMul<<<BLOCKS, THREADS>>>(a, b, c);
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c);

    std::cout << "PROGRAM COMPLETED SUCCESSFULLY!\n";
    
    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
