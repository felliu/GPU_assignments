#include <stdio.h>
#include <sys/time.h>
#include <limits>
#include <nppdefs.h>

#define TPB_DEFAULT 64
#define NUM_PARTICLES_DEFAULT 10000000
#define NUM_ITERATIONS_DEFAULT 1000

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void particleKernel(int num_particles, float3 *Particle, float3 *Velocity, int timeStep, float updateValue)
{

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_particles)
        return;

    Velocity[idx].x += updateValue;
    Velocity[idx].y += updateValue;
    Velocity[idx].z += updateValue;

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
    for (int i = 0; i < num_iters; ++i) {
        particleKernel<<<(num_particles + tpb - 1) / tpb, tpb>>>(num_particles, Particle, Velocity, i, updateValue);
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

    float3 *Particle_GPU;
    cudaMallocManaged(&Particle_GPU, num_particles * sizeof(float3));
    float3 *Velocity;
    cudaMallocManaged(&Velocity, num_particles * sizeof(float3));

    initFloat3(num_particles, Particle_GPU);
    initFloat3(num_particles, Velocity);

    float updateValue = 0.01;

    /*float3 *Particle_CPU = (float3*) malloc(num_particles * sizeof(float3));
    float3 *Velocity_CPU = (float3*) malloc(num_particles * sizeof(float3));


    memcpy(Particle_CPU, Particle_GPU, num_particles * sizeof(float3));
    memcpy(Velocity_CPU, Velocity, num_particles * sizeof(float3));
    
    //double cpuStart = cpuSecond();
    particleCPU(num_particles, num_iters, Particle_CPU, Velocity_CPU, updateValue);*/
    
    double cpuFinish = cpuSecond();
    particleGPU(num_particles, num_iters, Particle_GPU, Velocity, updateValue, tpb);
    cudaDeviceSynchronize();
    double gpuFinish = cpuSecond();
    //printf("CPU time: %f seconds\n", cpuFinish - cpuStart);
    printf("GPU time: %f seconds\n\n", gpuFinish - cpuFinish);

    //compareResult(num_particles, Particle_CPU, Particle_GPU);
    return 0;
}
