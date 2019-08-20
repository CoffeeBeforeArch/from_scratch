// This program computes a histogram on the GPU using CUDA
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <iostream>

using namespace std;

__global__ void histogram(int *input, int *bins, int N, int N_bins, int DIV){
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if(tid < N){
        int bin = input[tid] / DIV;
        atomicAdd(&bins[bin], 1);
    }

}

// Initialize our input array with random number from 0-99
void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
    }
}

int main(){
    // Number of elements to bin (2^20 default)
    int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    // Select the number of bins
    int N_bins = 10;
    size_t bytes_bins = N_bins * sizeof(int);

    // Allocate memory
    int *input, *bins;
    cudaMallocManaged(&input, bytes);
    cudaMallocManaged(&bins, bytes_bins);

    // Init our input array
    init_array(input, N);
    
    // Set the divisor for finding the corresponding bin for an input
    int DIV = (100 + N_bins - 1) / N_bins;

    // Set our bins = 0
    for(int i = 0; i < N_bins; i++){
        bins[i] = 0;
    }

    // Set the dimensions of our CTA and Grid
    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    histogram<<<BLOCKS, THREADS>>>(input, bins, N, N_bins, DIV);
    cudaDeviceSynchronize();

    return 0;
}
