#include <iostream>
#include <math.h>

#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 1000
#define BLOCK_SIZE 256

struct Particle {
    float3 position;
    float3 velocity;
};

__global__
void timestep(Particle *particles, int n, float iter_random)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float dt = 1.0;
    for (int i = index; i < n; i = i + stride){
        particles[i].velocity.x = -1*iter_random;
        particles[i].velocity.y = 0.5*iter_random;
        particles[i].velocity.z = 1*iter_random;
        particles[i].position.x = particles[i].position.x + dt*particles[i].velocity.x;
        particles[i].position.y = particles[i].position.y + dt*particles[i].velocity.y;
        particles[i].position.z = particles[i].position.z + dt*particles[i].velocity.z;
    }
}

void timestep_cpu(Particle *particles, int n, float iter_random){
    float dt = 1.0;
    for (int i = 0; i < n; i++){
        //printf("%d", i);
        particles[i].velocity.x = -1*iter_random;
        particles[i].velocity.y = 0.5*iter_random;
        particles[i].velocity.z = 1*iter_random;
        particles[i].position.x = particles[i].position.x + dt*particles[i].velocity.x;
        particles[i].position.y = particles[i].position.y + dt*particles[i].velocity.y;
        particles[i].position.z = particles[i].position.z + dt*particles[i].velocity.z;
    }
}



int main(){
    srand(1337);
    bool useGPU = false;
    Particle *particles;
    float *iter_randoms;
    cudaMallocManaged(&iter_randoms, NUM_ITERATIONS*sizeof(float));
    for(int i = 0; i < NUM_ITERATIONS; i++){
        iter_randoms[i] = rand();
    }
    cudaMallocManaged(&particles, NUM_PARTICLES*sizeof(Particle));
    if(useGPU){
        int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE;
        for(int iter = 0; iter < NUM_ITERATIONS; iter++){
            printf("iter %d", iter);
            timestep<<<numBlocks, BLOCK_SIZE>>>(particles, NUM_PARTICLES, iter_randoms[iter]);
            cudaDeviceSynchronize();
        }
        
    } else {
        for(int iter = 0; iter < NUM_ITERATIONS; iter++)
            timestep_cpu(particles, NUM_PARTICLES, iter_randoms[iter]);
    }
    for(int i = NUM_PARTICLES-5; i < NUM_PARTICLES; i++){
        printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    }
    cudaFree(iter_randoms);
    cudaFree(particles);
    return 0;
}