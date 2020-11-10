#include <iostream>
#include <math.h>
#include <windows.h>
#include <wchar.h>

#define NUM_PARTICLES 100000000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 128 //128 //256
#define NUM_DIMENSIONS 3

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

void init_random(Particle *particles, Particle *cpu_particles, int i){
    particles[i].position.x = (rand() % 10)/10.0 - 0.5;
    //printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    particles[i].position.y = (rand() % 10)/10.0 - 0.5;
    particles[i].position.z = (rand() % 10)/10.0 - 0.5;
    cpu_particles[i].position.x = particles[i].position.x;
    //printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    cpu_particles[i].position.y = particles[i].position.y;
    cpu_particles[i].position.z = particles[i].position.z;
}

int main(){
    wmain();
    srand(1337);
    //bool useGPU = true;
    Particle *particles;
    Particle *cpu_particles;
    float *iter_randoms;
    cudaMallocManaged(&iter_randoms, NUM_DIMENSIONS*NUM_ITERATIONS*sizeof(float));
    for(int i = 0; i < NUM_DIMENSIONS*NUM_ITERATIONS; i++){
        iter_randoms[i] = (rand() % 10)/10.0 - 0.5;
    }
    cudaMallocManaged(&particles, NUM_PARTICLES*sizeof(Particle));
    cudaMallocManaged(&cpu_particles, NUM_PARTICLES*sizeof(Particle));
    for(int i = 0; i < NUM_PARTICLES; i++){
        init_random(particles, cpu_particles, i);
    }
    printf("Before GPU\n");
    wmain();
    int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for(int iter = 0; iter < NUM_ITERATIONS; iter++){
        //printf("iter %d", iter);
        timestep<<<numBlocks, BLOCK_SIZE>>>(particles, NUM_PARTICLES, iter_randoms, iter);
        cudaDeviceSynchronize();
    }
    printf("After GPU\n");
    wmain();
    printf("Before CPU\n");
    wmain();
    //cudaDeviceSynchronize();
    for(int iter = 0; iter < NUM_ITERATIONS; iter++)
        timestep_cpu(cpu_particles, NUM_PARTICLES, iter_randoms, iter);
    printf("After CPU\n");
    wmain();
    float total_error = 0.0f;
    for(int i = 0; i < NUM_PARTICLES; i++){
        total_error += distance3d(particles[i].position, cpu_particles[i].position);
    }
    
    printf("total_error %f\n", total_error);
    for(int i = NUM_PARTICLES-5; i < NUM_PARTICLES; i++){
        printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
        printf("%f %f %f\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
    }
    printf("\n");
    for(int i = NUM_PARTICLES-5; i < NUM_PARTICLES; i++){
        printf("%f %f %f\n", cpu_particles[i].position.x, cpu_particles[i].position.y, cpu_particles[i].position.z);
        printf("%f %f %f\n", cpu_particles[i].velocity.x, cpu_particles[i].velocity.y, cpu_particles[i].velocity.z);
    }
    cudaFree(iter_randoms);
    cudaFree(particles);
    wmain();
    return 0;
}