#include <stdio.h>

#define N 256
#define TPB 256

__global__ void cuda_hello(){
    printf("Hello World! My threadId is %d\n", threadIdx.x);
}

int main() {
    cuda_hello<<<N/TPB,TPB>>>();
    cudaDeviceSynchronize();
    return 0;
}
