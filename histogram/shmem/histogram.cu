// This program computes a histogram using dynamically allocated shared
// memory and shared memory atomics
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <iostream>

using namespace std;

__global__ void histogram(int *input, int *bins, int N, int N_bins, int DIV){
    // Allocate shared memory
    extern __shared__ int s_bins[];

    // Calculate a global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize our shared memory
    if(threadIdx.x < N_bins)
        s_bins[threadIdx.x] = 0;

    // Wait for threads to zero out shared memory
    __syncthreads();

    // Range check
    if(tid < N){
        int bin = input[tid] / DIV;
        atomicAdd(&s_bins[bin], 1);
    }

    // Wait for threads to finish binning their elements
    __syncthreads();

    // Write back our partial results to main memory
    if(threadIdx.x < N_bins)
        atomicAdd(&bins[threadIdx.x], s_bins[threadIdx.x]);
}

// Initialize an array with random numbers
void init_array(int *input, int N, int MAX){
    for(int i = 0; i < N; i++){
        input[i] = rand() % MAX;
    }
}

int main(){
    // Set the number of elements to bin (2^10 default)
    int N = 1 << 12;
    size_t bytes = N * sizeof(int);

    // Set the number of bins
    int N_bins = 10;
    size_t bytes_bins = N_bins * sizeof(int);

    // Allocate memory
    int *input, *bins;
    cudaMallocManaged(&input, bytes);
    cudaMallocManaged(&bins, bytes_bins);

    // Set the max value for our data
    int MAX = 100;

    // Initialize our input data
    init_array(input, N, MAX);

    // Initialize our bins
    for(int i = 0; i < N_bins; i++){
        bins[i] = 0;
    }

    // Set the divisor
    int DIV = (MAX + N_bins - 1) / N_bins;

    // Set the CTA and Grid dimensions
    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Set the size of our dynamically allocated shmem
    size_t SHMEM = N_bins * sizeof(int);

    // Launch our kernel
    histogram<<<BLOCKS, THREADS, SHMEM>>>(input, bins, N, N_bins, DIV);
    cudaDeviceSynchronize();

    return 0;
}
