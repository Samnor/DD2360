#include <iostream>
#include <math.h>
#include <windows.h>
#include <wchar.h>

#define NUM_PARTICLES 1000000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 256 //128 //256
#define NUM_DIMENSIONS 3
#define USE_PINNED_MEMORY 1

typedef struct Particle {
    float3 position;
    float3 velocity;
} Particle;

int wmain(void) {

    SYSTEMTIME lt = {0};
  
    GetLocalTime(&lt);
  
    wprintf(L"The local time is: %02d:%02d:%02d:%04d\n", 
        lt.wHour, lt.wMinute, lt.wSecond, lt.wMilliseconds);

    return 0;
}

float distance1d(float x1, float x2)
{
    return sqrt((x2 - x1)*(x2 - x1));
}

float distance3d(float3 x1, float3 x2){
    float distance_sum = 0.0f;
    distance_sum += distance1d(x1.x, x2.x);
    distance_sum += distance1d(x1.y, x2.y);
    distance_sum += distance1d(x1.z, x2.z);
    return distance_sum;
}

__global__
void timestep(Particle *particles, int n, float *iter_randoms, int iter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float dt = 1.0;
    int iter_index = iter*3;
    for (int i = index; i < n; i = i + stride){
        particles[i].velocity.x = iter_randoms[iter_index];
        particles[i].velocity.y = iter_randoms[iter_index+1];
        particles[i].velocity.z = iter_randoms[iter_index+2];
        particles[i].position.x = particles[i].position.x + dt*particles[i].velocity.x;
        particles[i].position.y = particles[i].position.y + dt*particles[i].velocity.y;
        particles[i].position.z = particles[i].position.z + dt*particles[i].velocity.z;
    }
}

void timestep_cpu(Particle *particles, int n, float *iter_randoms, int iter){
    float dt = 1.0;
    int iter_index = iter*3;
    for (int i = 0; i < n; i++){
        //printf("%d", i);
        particles[i].velocity.x = iter_randoms[iter_index];
        particles[i].velocity.y = iter_randoms[iter_index+1];
        particles[i].velocity.z = iter_randoms[iter_index+2];
        particles[i].position.x = particles[i].position.x + dt*particles[i].velocity.x;
        particles[i].position.y = particles[i].position.y + dt*particles[i].velocity.y;
        particles[i].position.z = particles[i].position.z + dt*particles[i].velocity.z;
    }
}

void init_random(Particle *cpu_particles, int i){
    cpu_particles[i].position.x = (rand() % 10)/10.0 - 0.5;
    //printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    cpu_particles[i].position.y = (rand() % 10)/10.0 - 0.5;
    cpu_particles[i].position.z = (rand() % 10)/10.0 - 0.5;
}

int main(){
    /*
    Modify the program, such that
    All particles are copied to the GPU at the beginning of a time step.
    All the particles are copied back to the host after the kernel completes, before proceeding to the next time step.
    */

    printf("start of main()\n");
    wmain();
    srand(1337);
    // Initiate data structures
    Particle *cpu_particles;
    Particle *gpu_particles;
    //cpu_particles = (Particle *)calloc(NUM_PARTICLES, sizeof(Particle));
    //cudaMallocHost(&cpu_particles, NUM_PARTICLES*sizeof(Particle));
    cudaMalloc((void **) &gpu_particles, NUM_PARTICLES*sizeof(Particle));
    if(USE_PINNED_MEMORY){
        cudaHostAlloc((void **) &cpu_particles, NUM_PARTICLES*sizeof(Particle), cudaHostAllocDefault);
    } else {
        cpu_particles = (Particle *)calloc(NUM_PARTICLES, sizeof(Particle));
    }
    
    // Initiate random data on GPU
    float *iter_randoms;
    float *gpu_iter_randoms;
    iter_randoms = (float *)calloc(NUM_DIMENSIONS*NUM_ITERATIONS, sizeof(float));
    //cudaMalloc(&iter_randoms, NUM_PARTICLES*sizeof(float));
    for(int i = 0; i < NUM_DIMENSIONS*NUM_ITERATIONS; i++){
        iter_randoms[i] = (rand() % 10)/10.0 - 0.5;
    }
    for(int i = 0; i < NUM_PARTICLES; i++){
        init_random(cpu_particles, i);
    }
    cudaMalloc(&gpu_iter_randoms, NUM_DIMENSIONS*NUM_ITERATIONS * sizeof(Particle));
    printf("Before GPU\n");
    wmain();
    int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Run timesteps
    for(int t = 0; t < NUM_ITERATIONS; t++){
        cudaMemcpy(gpu_particles, cpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
        printf("t %d\n", t);
        timestep<<<numBlocks, BLOCK_SIZE>>>(gpu_particles, NUM_PARTICLES, gpu_iter_randoms, t);
        cudaMemcpy(cpu_particles, gpu_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    printf("After GPU\n");
    wmain();

    cudaFree(gpu_particles);
    cudaFree(gpu_iter_randoms);
    if(USE_PINNED_MEMORY){
        cudaFree(cpu_particles);
    } else {
        free(cpu_particles);
    }
    free(iter_randoms);
    cudaDeviceSynchronize();
    printf("end of program\n");
    return 0;
}
