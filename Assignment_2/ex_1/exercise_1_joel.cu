#include <stdio.h>

__global__ void helloWorld() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World! My ThreadId is %2d\n", i);
}

int main() {
    helloWorld<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}
