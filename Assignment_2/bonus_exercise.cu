#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/time.h>

#define SEED     921
#define NUM_ITER 10000000 // Iterations per thread
#define NUM_BLOCKS 1024
#define TPB 128


__global__ void calculatePi(curandState *dev_random, unsigned long long *totals) {
    __shared__ unsigned long long count[NUM_BLOCKS];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(SEED, i, 0, &dev_random[i]);
    double x, y, z;
    count[threadIdx.x] = 0;
    for (int iter = 0; iter < NUM_ITER; ++iter) {
        x = curand_uniform(&dev_random[i]);
        y = curand_uniform(&dev_random[i]);
        z = (x*x) + (y*y);

        if (z <= 1.0)
        {
            count[threadIdx.x] += 1;
        }
    }

    if (threadIdx.x == 0) {
        totals[blockIdx.x] = 0;
        for (int i = 0; i < TPB; ++i) {
            totals[blockIdx.x] += count[i];
        }
    }
}


int main(int argc, char* argv[]) {
    struct timeval start, end;
    curandState *dev_random;
    unsigned long long *totals, *d_totals;
    unsigned long long NumThreads = (unsigned long long) (NUM_BLOCKS * TPB);
    unsigned long long NumIter = (double) NUM_ITER;
    cudaMalloc((void**)&dev_random, NumThreads * sizeof(curandState));
    cudaMalloc(&d_totals, NUM_BLOCKS * sizeof(unsigned long long));
    totals = (unsigned long long*)malloc(NUM_BLOCKS * sizeof(unsigned long long));
    
    gettimeofday(&start, NULL);
    calculatePi<<<NUM_BLOCKS, TPB>>>(dev_random, d_totals);
    gettimeofday(&end, NULL);

    cudaMemcpy(totals, d_totals, NUM_BLOCKS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long count = 0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        count += totals[i];
    }
    double pi = ((double) count / (double)(NumThreads * NumIter)) * 4.0;
    printf(
        "The result is %.15f after %ld samples in %ld microseconds!\n",
        pi,
        (long int)(NumThreads * NumIter),
        ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));

    cudaFree(dev_random);
    cudaFree(d_totals);
    cudaFree(totals);

    return 0;
}