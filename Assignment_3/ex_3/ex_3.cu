#include <stdio.h>
#include <sys/time.h>
#include <limits>
#include <nppdefs.h>

#define TPB_DEFAULT 64
#define NUM_PARTICLES_DEFAULT 10000000
#define NUM_ITERATIONS_DEFAULT 1000
#define BATCH_SIZE 1000
#define NUM_STREAMS 4
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void particleKernel(int num_particles, int batch_offset, float3 *Particle, float3 *Velocity, float updateValue)
{

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_particles)
        return;

    Velocity[batch_offset + idx].x += updateValue;
    Velocity[batch_offset + idx].y += updateValue;
    Velocity[batch_offset + idx].z += updateValue;

    Particle[idx].x += Velocity[idx].x;
    Particle[idx].y += Velocity[idx].y;
    Particle[idx].z += Velocity[idx].z;
}

void particleCPU(int num_particles, int num_iters, float3 *Particle, float3 *Velocity, float updateValue)
{
    for (int i = 0; i < num_iters; i++)
        for (int j = 0; j < num_particles; j++)
        {
            Velocity[j].x += updateValue;
            Velocity[j].y += updateValue;
            Velocity[j].z += updateValue;

            Particle[j].x += Velocity[j].x;
            Particle[j].y += Velocity[j].y;
            Particle[j].z += Velocity[j].z;
        }
}

void particleGPU(int num_particles, int num_iters, float3 *Particle, float3 *Velocity, float updateValue, int tpb)
{
    //Create one device buffer per stream to overlap
    float3* particle_buffers[NUM_STREAMS] = {0};
    float3 *d_velocity;

    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&particle_buffers[i], BATCH_SIZE * sizeof(float3));
    }

    cudaMalloc(&d_velocity, num_particles * sizeof(float3));
    cudaMemcpy(d_velocity, Velocity, num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    const int num_batches = (num_particles + (BATCH_SIZE - 1)) / BATCH_SIZE;

    for (int iter = 0; iter < num_iters; ++iter) {
        bool more_batches = true;
        int batch_idx = 0;

        while(more_batches) {
            for (int i = 0; i < NUM_STREAMS; ++i) {
                if (batch_idx >= num_batches) {
                    more_batches = false;
                    break;
                }
                //Check if we are at the final batch, which may be a smaller size
                const int curr_batch_size = (batch_idx == num_batches - 1) ? num_particles - batch_idx * BATCH_SIZE : BATCH_SIZE;
                cudaMemcpyAsync(particle_buffers[i], Particle + batch_idx * BATCH_SIZE, curr_batch_size * sizeof(float3),
                                cudaMemcpyHostToDevice, streams[i]);
                particleKernel<<<(curr_batch_size + tpb - 1) / tpb,
                                  tpb, 0, streams[i]>>>(curr_batch_size, batch_idx * BATCH_SIZE, particle_buffers[i], d_velocity, updateValue);
                cudaMemcpyAsync(Particle + batch_idx * BATCH_SIZE, particle_buffers[i], curr_batch_size * sizeof(float3), cudaMemcpyDeviceToHost, streams[i]);
                ++batch_idx;
            }
        }
        //Make sure all streams finished
        cudaDeviceSynchronize();
    }
}

void initFloat3(int n, float3 *input)
{

    for (int i = 0; i < n; i++)
    {
        input[i].x = (float(rand()) / float((RAND_MAX)));
        input[i].y = (float(rand()) / float((RAND_MAX)));
        input[i].z = (float(rand()) / float((RAND_MAX)));
    }
}

void compareResult(int num_particles, float3 *Particle_CPU, float3 *Particle_GPU)
{

    for (int i = 0; i < num_particles; i++)
    {

        if (fabs(Particle_CPU[i].x - Particle_GPU[i].x) > 1e-6 ||
            fabs(Particle_CPU[i].y - Particle_GPU[i].y) > 1e-6 ||
            fabs(Particle_CPU[i].z - Particle_GPU[i].z) > 1e-6)
        {
            printf("Difference at %d particle : \n", i);
            printf("CPU particle.x = %f, CPU particle.y = %f,"
                   "CPU particle.z = %f, GPU particle.x = %f,"
                   "GPU particle.y = %f, GPU particle.z = %f \n",
                   Particle_CPU[i].x, Particle_CPU[i].y, Particle_CPU[i].z, Particle_GPU[i].x, Particle_GPU[i].y, Particle_GPU[i].z);
        }
    }
}

int main(int argc, char *argv[])
{
    int tpb = TPB_DEFAULT;
    int num_particles = NUM_PARTICLES_DEFAULT;
    int num_iters = NUM_ITERATIONS_DEFAULT;

    if (argc == 4) {
        num_particles = atoi(argv[1]);
        num_iters = atoi(argv[2]);
        tpb = atoi(argv[3]);
    }

    printf("Config: num_particles: %d, num_iters: %d, tpb: %d\n", num_particles, num_iters, tpb);

    float3 *Particle_CPU = (float3*) malloc(num_particles * sizeof(float3));
    float3 *Particle_GPU;
    cudaMallocHost(&Particle_GPU, num_particles * sizeof(float3));
    float3 *Velocity;
    cudaMallocHost(&Velocity, num_particles * sizeof(float3));
    float3 *Velocity_CPU = (float3*) malloc(num_particles * sizeof(float3));

    initFloat3(num_particles, Particle_CPU);
    initFloat3(num_particles, Velocity);
    cudaMemcpy(Particle_GPU, Particle_CPU, num_particles * sizeof(float3), cudaMemcpyHostToHost);
    cudaMemcpy(Velocity_CPU, Velocity, num_particles * sizeof(float3), cudaMemcpyHostToHost);
    float updateValue = 0.01;
    
    /*double cpuStart = cpuSecond();
    particleCPU(num_particles, num_iters, Particle_CPU, Velocity_CPU, updateValue);*/
    double cpuFinish = cpuSecond();
    particleGPU(num_particles, num_iters, Particle_GPU, Velocity, updateValue, tpb);
    double gpuFinish = cpuSecond();
    //printf("CPU time: %f seconds\n", cpuFinish - cpuStart);
    printf("GPU time: %f seconds\n\n", gpuFinish - cpuFinish);

    //compareResult(num_particles, Particle_CPU, Particle_GPU);
    return 0;
}
