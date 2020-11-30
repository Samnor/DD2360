#include <iostream>
#include <math.h>
#include <windows.h>
#include <wchar.h>

#define NUM_PARTICLES 1000000
#define NUM_ITERATIONS 1000
#define BLOCK_SIZE 256 //128 //256
#define BATCH_SIZE 32
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
void timestep_stream(Particle *particles, int offset, int stream_n, int iter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float dt = 1.0;
    //int iter_index = iter*3;
    for (int i = index + offset; i < stream_n; i = i + stride){
        //particles[i].velocity.x = iter_randoms[iter_index];
        //particles[i].velocity.y = iter_randoms[iter_index+1];
        //particles[i].velocity.z = iter_randoms[iter_index+2];
        particles[i].position.x = particles[i].position.x + dt*particles[i].velocity.x;
        particles[i].position.y = particles[i].position.y + dt*particles[i].velocity.y;
        particles[i].position.z = particles[i].position.z + dt*particles[i].velocity.z;
    }
}

/*
__global__
void timestep_stream(Particle *particles, int offset, int n, int iter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + offset;
    float dt = 1.0;
    //printf("Particles before %f %f %f\n", particles[index].position.x, particles[index].position.y, particles[index].position.z);
    particles[index].position.x = particles[index].position.x + dt*particles[index].velocity.x;
    particles[index].position.y = particles[index].position.y + dt*particles[index].velocity.y;
    particles[index].position.z = particles[index].position.z + dt*particles[index].velocity.z;

    //printf("Particles after %f %f %f\n", particles[index].position.x, particles[index].position.y, particles[index].position.z);
}
*/

void init_random(Particle *cpu_particles, int i){
    cpu_particles[i].position.x = (rand() % 10)/10.0 - 0.5;
    //printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    cpu_particles[i].position.y = (rand() % 10)/10.0 - 0.5;
    cpu_particles[i].position.z = (rand() % 10)/10.0 - 0.5;
}

void init_velocity(Particle *cpu_particles, float *iter_randoms){
    for(int i = 0; i < NUM_ITERATIONS*NUM_DIMENSIONS; i++){
        cpu_particles[i].velocity.x = (rand() % 10)/10.0 - 0.5;
        cpu_particles[i].velocity.y = (rand() % 10)/10.0 - 0.5;
        cpu_particles[i].velocity.z = (rand() % 10)/10.0 - 0.5;
    }
}

void print_velocity(Particle *cpu_particles){
    for(int i = 0; i < NUM_ITERATIONS*NUM_DIMENSIONS; i++){
        printf("%f %f %f\n", cpu_particles[i].velocity.x, cpu_particles[i].velocity.y, cpu_particles[i].velocity.z);
    }
}

void print_particles(Particle *cpu_particles, int num_to_print){
    printf("print_particles()\n");
    for(int i = 0; i < num_to_print; i++){
        printf("%f %f %f\n", cpu_particles[i].position.x, cpu_particles[i].position.y, cpu_particles[i].position.z);
    }
}

void run_particle_iterations(Particle *cpu_particles, Particle *gpu_particles){
    const int num_streams = 4;
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    const int stream_particle_count = NUM_PARTICLES/num_streams;
    const int batch_size = 1000;//stream_particle_count;
    const int stream_loops = (NUM_PARTICLES + batch_size - 1)/batch_size;
    printf("stream_loops %d \n", stream_loops);
    printf("batch_size %d\n", batch_size);
    //const int stream_bytes = stream_particle_count * sizeof(Particle);
    const int batch_bytes = batch_size * sizeof(Particle);
    for(int t = 0; t < NUM_ITERATIONS; t++){
        //printf("t: %d\n", t);
        //print_particles(cpu_particles, 1);
        for (int i = 0; i < stream_loops; ++i) {
            //printf("start of stream loop %d\n", i);
            int stream_index = i % num_streams;
            cudaStreamSynchronize(streams[stream_index]);
            //printf("stream_index %d\n", stream_index);
            int offset = i * stream_particle_count;
            //cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&gpu_particles[offset], &cpu_particles[offset], batch_bytes, cudaMemcpyHostToDevice, streams[stream_index]);
            int num_blocks = (stream_particle_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timestep_stream<<<num_blocks, BLOCK_SIZE, 0, streams[stream_index]>>>(gpu_particles, offset, batch_size, t);
            //kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
            //cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
            cudaMemcpyAsync(&cpu_particles[offset], &gpu_particles[offset], batch_bytes, cudaMemcpyDeviceToHost, streams[stream_index]);
        }

        /*
        print_particles(cpu_particles, 1);
        for (int i = 0; i < num_streams; ++i) {
            int offset = i * stream_particle_count;
            cudaMemcpyAsync(&gpu_particles[offset], &cpu_particles[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]);
        }

        for (int i = 0; i < num_streams; ++i) {
            int offset = i * stream_particle_count;
            //kernel<<<stream_particle_count/blockSize, blockSize, 0, stream[i]>>>(gpu_particles, offset);
            int num_blocks = (stream_particle_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timestep_stream<<<num_blocks, BLOCK_SIZE, 0, streams[i]>>>(gpu_particles, offset, stream_particle_count, t);
        }
        printf("before transfer\n");
        for (int i = 0; i < num_streams; ++i) {
            //cudaStreamSynchronize(streams[i]);
            int offset = i * stream_particle_count;
            cudaMemcpyAsync(&cpu_particles[offset], &gpu_particles[offset], stream_bytes, cudaMemcpyDeviceToHost, streams[i]);
        }
        //cudaDeviceSynchronize();
        printf("After transfer\n");
        //print_particles(cpu_particles, 1);
        */
    }
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

void init_particles_and_velocity(float *iter_randoms, Particle *cpu_particles){
    for(int i = 0; i < NUM_DIMENSIONS*NUM_ITERATIONS; i++){
        iter_randoms[i] = (rand() % 10)/10.0 - 0.5;
        printf("iter_randoms[i] %f\n", iter_randoms[i]);
    }
    for(int i = 0; i < NUM_PARTICLES; i++){
        init_random(cpu_particles, i);
    }
    init_velocity(cpu_particles, iter_randoms);
}

int main(){
    printf("start of main()\n");
    wmain();
    srand(1337);
    // Initiate data structures
    Particle *cpu_particles;
    Particle *gpu_particles;
    cudaMalloc((void **) &gpu_particles, NUM_PARTICLES*sizeof(Particle));
    cudaHostAlloc((void **) &cpu_particles, NUM_PARTICLES*sizeof(Particle), cudaHostAllocDefault);

    // Initiate random data on GPU
    float *iter_randoms;
    iter_randoms = (float *)calloc(NUM_DIMENSIONS*NUM_ITERATIONS, sizeof(float));
    //cudaMalloc(&iter_randoms, NUM_PARTICLES*sizeof(float));
    printf("init iter_randoms\n");
    printf("Before GPU\n");
    wmain();
    init_particles_and_velocity(iter_randoms, cpu_particles);
    //print_velocity(cpu_particles);
    run_particle_iterations(cpu_particles, gpu_particles);
    print_particles(cpu_particles, 20);
    printf("After loop of batches\n");
    //print_particles(cpu_particles, NUM_PARTICLES);
    printf("after print_particles in end\n");
    //cudaStreamSynchronize(stream1); // wait for just one stream to finish all activites
    //cudaDeviceSynchronize();
    for(int i = NUM_PARTICLES-5; i < NUM_PARTICLES; i++){
        printf("%f %f %f\n", cpu_particles[i].position.x, cpu_particles[i].position.y, cpu_particles[i].position.z);
        //printf("%f %f %f\n", cpu_particles[i].velocity.x, cpu_particles[i].velocity.y, cpu_particles[i].velocity.z);
    }
    
    cudaFree(cpu_particles);
    cudaFree(gpu_particles);
}
