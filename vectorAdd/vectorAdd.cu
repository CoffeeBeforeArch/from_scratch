// This program computes the sum of two arrays on the GPU using CUDA
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

// Computes the sum of two arrays
__global__ void vectorAdd(int *a, int *b, int *c, int N){
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Range check
    if(tid < N)
        c[tid] = a[tid] + b[tid];

}

// Initializes an array of size "N" with numbers between 0 and 100
void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
    }
}

// Verify the vector add computation on the CPU
void verify_solution(int *a, int *b, int *c, int N){
    for(int i = 0; i < N; i++){
        assert(a[i] + b[i] == c[i]);
    } 
}

int main(){
    // Set our problem size (Default = 2^20)
    int N = 1 << 20;
    size_t bytes = N * sizeof(bytes);

    // Allocate some memory for our inputs/outputs
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize our data
    init_array(a, N);
    init_array(b, N);

    // Initialize our CTA and Grid dimensions
    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Call the kernel
    vectorAdd<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Verify the solution
    verify_solution(a, b, c, N);

    cout << "PROGRAM COMPLETED CORRECTLY!" << endl;
    
    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

