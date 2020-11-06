
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < n; i = i + stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int ARRAY_SIZE = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, ARRAY_SIZE*sizeof(float));
  cudaMallocManaged(&y, ARRAY_SIZE*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < ARRAY_SIZE; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (ARRAY_SIZE + blockSize - 1) / blockSize; // Forces numBlocks to always have enough blocks to handle all elements event if the last block is not fully utilized
  add<<<numBlocks, blockSize>>>(ARRAY_SIZE, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < ARRAY_SIZE; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}