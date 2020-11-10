#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#define NUM_PARTICLES 10000 // Third argument
#define NUM_ITERATIONS 100 // Second argument
#define BLOCK_SIZE 16 // First argument

typedef struct
{
   float3 position;
   float3 velocity;
} Particle;


__global__ void timeStep(Particle *particles, int time) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_PARTICLES) {
        particles[i].velocity.x = (time % 3 == 0) ? time % 5 / 1.7 : -time % 5 / 2.8;
        particles[i].velocity.y = (time % 7 == 0) ? time % 4 / 4.5: -time % 3 / 2.3;
        particles[i].velocity.z = (time % 2 == 0) ? time % 3 * 1.6 : -time % 7 / 1.2;
        particles[i].position.x += particles[i].velocity.x;
        particles[i].position.y += particles[i].velocity.y;
        particles[i].position.z += particles[i].velocity.z;
    }
}

void timeStepCPU(Particle *particles, int time) {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles[i].velocity.x = (time % 3 == 0) ? time % 5 / 1.7 : -time % 5 / 2.8;
        particles[i].velocity.y = (time % 7 == 0) ? time % 4 / 4.5: -time % 3 / 2.3;
        particles[i].velocity.z = (time % 2 == 0) ? time % 3 * 1.6 : -time % 7 / 1.2;
        particles[i].position.x += particles[i].velocity.x;
        particles[i].position.y += particles[i].velocity.y;
        particles[i].position.z += particles[i].velocity.z;
    }
}

int main(int argc, char *argv[]){
    int numParticles, numIterations, blockSize;
    if (argc == 1) {
        numParticles = NUM_PARTICLES;
        numIterations = NUM_ITERATIONS;
        blockSize = BLOCK_SIZE;
    } else if (argc == 2) {
        numParticles = NUM_PARTICLES;
        numIterations = NUM_ITERATIONS;
        blockSize = atoi(argv[1]);
    } else if (argc == 3) {
        numParticles = NUM_PARTICLES;
        numIterations = atoi(argv[2]);
        blockSize = atoi(argv[1]);
    } else {
        numParticles = atoi(argv[3]);
        numIterations = atoi(argv[2]);
        blockSize = atoi(argv[1]);
    }
    //struct timeval start, end;

    Particle *particles = (Particle*)malloc(numParticles * sizeof(Particle));
    Particle *particles_GPU = (Particle*)malloc(numParticles * sizeof(Particle));
    Particle *d_particles;
    for (int i = 0; i < numParticles; ++i) {
        particles[i].velocity.x = 1;
        particles[i].velocity.y = 1;
        particles[i].velocity.z = 1;
        particles[i].position.x = 0;
        particles[i].position.y = 0;
        particles[i].position.z = 0;
    }
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    //gettimeofday(&start, NULL);
    for (int t = 0; t < numIterations; ++t){
        timeStepCPU(particles, t);
    }
    //gettimeofday(&end, NULL);
    //printf("Computing %d iterations on the CPU... Done in %ld microseconds!\n", numIterations, ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
    
    //gettimeofday(&start, NULL);
    for (int t = 0; t < numIterations; ++t){
        timeStep<<<blockSize, numParticles / blockSize + 1>>>(d_particles, t);
    }
    //gettimeofday(&end, NULL);
    //printf("Computing %d iterations on the GPU... Done in %ld microseconds!\n", numIterations, ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
    cudaMemcpy(particles_GPU, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    int sameArray = 1;
    for (int i = 0; i < numParticles; ++i) {
        if ((particles[i].position.x != particles_GPU[i].position.x) || (particles[i].position.y != particles_GPU[i].position.y)|| (particles[i].position.z != particles_GPU[i].position.z)) {
            printf("Comparing the output for each implementation… Wrong at %d!\n", i);
            printf("GPU.x: %f, CPU.x: %f\n", particles[i].position.x, particles_GPU[i].position.x);
            sameArray = 0;
            break;
        }
    }
    if (sameArray == 1) {
        printf("Comparing the output for each implementation… Correct!\n");
    }


    free(particles);
    free(particles_GPU);
    cudaFree(d_particles);
}