// This program shows off global synchronization in CUDA using Cooperative
// Groups
// By: Nick from CoffeeBeforeArch

#include <cooperative_groups.h>
#include <iostream>

using std::cout;
using std::endl;
namespace cg = cooperative_groups;

__global__ void globalSync(int *a_1, int *a_2, int N) {
  // Calculate a global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Get the grid we want to synchronize
  auto g = cg::this_grid();

  // Fill the a_1 array with increasing numbers
  for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
    a_1[i] = i;
  }

  // Synchronize the entire grid
  g.sync();

  // Each thread does the sum of the entire a_1 array
  for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
    // Use a temporary to store the sum
    int temp = 0;
    for (int j = 0; j < N; j++) {
      temp += a_1[j];
    }

    // Write back the final result
    a_2[i] = temp;
  }
}

int main() {
  // Define our problem size
  int N = 1 << 10;
  size_t bytes = N * sizeof(int);

  // Allocate memory using unified memory
  int *a_1, *a_2;
  cudaMallocManaged(&a_1, bytes);
  cudaMallocManaged(&a_2, bytes);

  // Set our Grid and CTA sizes
  int THREADS;
  int BLOCKS;
  cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, globalSync, 0, 0);

  // Call our kernel
  globalSync<<<BLOCKS, THREADS>>>(a_1, a_2, N);
  cudaDeviceSynchronize();

  return 0;
}
