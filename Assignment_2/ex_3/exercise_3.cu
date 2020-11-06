#include <iostream>
#include <math.h>


#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 10000
#define BLOCK_SIZE 256
#define NUM_DIMENSIONS 3

typedef struct Particle {
    float3 position;
    float3 velocity;
} Particle;

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

void init_random(Particle *particles, int i){
    particles[i].position.x = (rand() % 10)/10.0 - 0.5;
    //printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    particles[i].position.y = (rand() % 10)/10.0 - 0.5;
    particles[i].position.z = (rand() % 10)/10.0 - 0.5;
}

int main(){
    srand(1337);
    bool useGPU = false;
    Particle *particles;
    float *iter_randoms;
    cudaMallocManaged(&iter_randoms, NUM_DIMENSIONS*NUM_ITERATIONS*sizeof(float));
    for(int i = 0; i < NUM_DIMENSIONS*NUM_ITERATIONS; i++){
        iter_randoms[i] = (rand() % 10)/10.0 - 0.5;
    }
    cudaMallocManaged(&particles, NUM_PARTICLES*sizeof(Particle));
    for(int i = 0; i < NUM_PARTICLES; i++){
        init_random(particles, i);
    }
    
    if(useGPU){
        int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE;
        for(int iter = 0; iter < NUM_ITERATIONS; iter++){
            //printf("iter %d", iter);
            timestep<<<numBlocks, BLOCK_SIZE>>>(particles, NUM_PARTICLES, iter_randoms, iter);
            
        }
        cudaDeviceSynchronize();
        
    } else {
        for(int iter = 0; iter < NUM_ITERATIONS; iter++)
            timestep_cpu(particles, NUM_PARTICLES, iter_randoms, iter);
    }
    printf("last print\n");
    for(int i = NUM_PARTICLES-5; i < NUM_PARTICLES; i++){
        printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
        printf("%f %f %f\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
    }
    for(int i = NUM_PARTICLES-5; i < NUM_PARTICLES; i++){
        printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
        printf("%f %f %f\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
    }
    cudaFree(iter_randoms);
    cudaFree(particles);
    return 0;
}