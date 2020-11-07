#include <stdio.h>
#include <sys/time.h>
#define ARRAY_SIZE 1000000
#define TPB 256

void saxpy_cpu(float *y, float *x, const float a) {
    for (int i = 0; i < ARRAY_SIZE; ++i){
        y[i] = a * x[i] + y[i];
    }
}


__global__ void saxpy(float *y, float *x, const float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ARRAY_SIZE) {
        y[i] = a * x[i] + y[i];
    }
}


int main(void) {
    float *y = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *x = (float*)malloc(ARRAY_SIZE * sizeof(float));;
    const float a = 2;
    for (int i = 0; i < ARRAY_SIZE; ++i){
        y[i] = 2.0f;
        x[i] = 1.0f;
    }
    struct timeval start, end;
    gettimeofday(&start, NULL);
    saxpy_cpu(y, x, a);
    gettimeofday(&end, NULL);
    printf("Computing SAXPY on the CPU... Done in %ld microseconds!\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));

    float *y_CPU = (float*)malloc(ARRAY_SIZE * sizeof(float));
    for (int i = 0; i < ARRAY_SIZE; ++i){
        y_CPU[i] = y[i];
        y[i] = 2.0f;
    }

    float *d_y;
    float *d_x;
    cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));

    cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);
    saxpy<<<(ARRAY_SIZE + TPB - 1)/TPB, TPB>>>(d_y, d_x, a);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("Computing SAXPY on the GPU... Done in %ld microseconds!\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));

    cudaMemcpy(y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


    int sameArray = 1;
    for (int i = 0; i < ARRAY_SIZE; ++i){
        if (y[i] != y_CPU[i]) {
            printf("Comparing the output for each implementation… Wrong!\n");
            sameArray = 0;
            break;
        }
    }
    if (sameArray == 1) {
        printf("Comparing the output for each implementation… Correct!\n");
    }

    free(y);
    free(x);
    free(y_CPU);
    cudaFree(d_y);
    cudaFree(d_x);

}
