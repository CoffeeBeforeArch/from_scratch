// This program computes 1D convolution on the GPU using constant memory
// By: Nick from CoffeeBeforeArch

#include <cstdlib>

#define MASK_LEN 7

// Allocate constant memory
__constant__ int MASK[MASK_LEN];

__global__ void convolution(int *input, int *output, int N){
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if(tid < N){
        int radius = MASK_LEN / 2;
        int start = tid - radius;
        int tmp = 0;
        // Iterate over the length of the mask
        for(int i = 0; i < MASK_LEN; i++){
            // Check if we are off either edge of the array
            if(start + i >= 0 && start + i < N){
                // Accumulate each product
                tmp += MASK[i] * input[start + i];
            }
        }

        // Write back the result
        output[tid] = tmp;
    }

}

// Initializes an array with random numbers
void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
    }
}

int main(){
    // Set the input array size (2^20 default)
    int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    // Allocate space for our input and output
    int *input, *output;
    cudaMallocManaged(&input, bytes);
    cudaMallocManaged(&output, bytes);

    // Initialize our input array
    init_array(input, N);

    // Allocate space for the mask on the host
    int *mask = new int[MASK_LEN];
    init_array(mask, MASK_LEN);

    // Copy the mask to the GPU
    cudaMemcpyToSymbol(MASK, mask, MASK_LEN * sizeof(int));

    // Set our CTA and Grid dimensions
    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Launch our kernel
    convolution<<<BLOCKS, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();

    return 0;
}
