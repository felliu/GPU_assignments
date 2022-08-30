#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std::chrono;

typedef struct __attribute__ ((packed)) Particle {
    float pos_x, pos_y, pos_z;
    float vel_x, vel_y, vel_z;
} Particle;

void init_particles(std::vector<Particle>& particles) {
    for (Particle& particle : particles) {
        particle.pos_x = std::rand() / static_cast<float>(RAND_MAX);
        particle.pos_y = std::rand() / static_cast<float>(RAND_MAX);
        particle.pos_z = std::rand() / static_cast<float>(RAND_MAX);

        particle.vel_x = std::rand() / static_cast<float>(RAND_MAX);
        particle.vel_y = std::rand() / static_cast<float>(RAND_MAX);
        particle.vel_z = std::rand() / static_cast<float>(RAND_MAX);
    }
}

const char* update_particles_str =
"typedef struct __attribute__ ((packed)) Particle {\n"
"   float pos_x, pos_y, pos_z;\n"
"   float vel_x, vel_y, vel_z;\n"
"} Particle;                  \n"
"__kernel void update_particles(int num_particles, __global Particle *particles, float velocity_update) {\n"
"    size_t i = get_global_id(0);                \n"
"    if (i >= num_particles) return;          \n"
"    particles[i].vel_x += velocity_update;   \n"
"    particles[i].vel_y += velocity_update;   \n"
"    particles[i].vel_z += velocity_update;   \n"
"                                             \n"
"    particles[i].pos_x += particles[i].vel_x;\n"
"    particles[i].pos_y += particles[i].vel_y;\n"
"    particles[i].pos_z += particles[i].vel_z;\n"
"}";

size_t update_particles_cpu(int iterations, std::vector<Particle>& particles, float velocity_update) {
    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (int j = 0; j < particles.size(); ++j) {
            particles[j].vel_x += velocity_update;
            particles[j].vel_y += velocity_update;
            particles[j].vel_z += velocity_update;

            particles[j].pos_x += particles[j].vel_x;
            particles[j].pos_y += particles[j].vel_y;
            particles[j].pos_z += particles[j].vel_z;
        }
    }
    auto end_time = high_resolution_clock::now();
    auto duration_count = duration_cast<milliseconds>(end_time - start_time).count();
    return duration_count;
}

bool compare_results(const std::vector<Particle>& cpu_particles, const std::vector<Particle>& gpu_particles) {
    for (int i = 0; i < cpu_particles.size(); ++i) {
        if (std::fabs(cpu_particles[i].pos_x - gpu_particles[i].pos_x > 1e-6) ||
            std::fabs(cpu_particles[i].pos_y - gpu_particles[i].pos_y > 1e-6) ||
            std::fabs(cpu_particles[i].pos_z - gpu_particles[i].pos_z > 1e-6)) {
            
            std::cerr << "Particles differ at position: " << i << "\n";
            std::cerr << "cpu_x: " << cpu_particles[i].pos_x << " gpu_x: " << gpu_particles[i].pos_x << "\n";
            std::cerr << "cpu_y: " << cpu_particles[i].pos_y << " gpu_y: " << gpu_particles[i].pos_y << "\n";
            std::cerr << "cpu_z: " << cpu_particles[i].pos_z << " gpu_z: " << gpu_particles[i].pos_z << "\n";
            return false;
        }
    }

    return true;
}

size_t run_gpu_sim(int iterations, int block_size, std::vector<Particle>& particles, float velocity_update) {
    const size_t num_particles = particles.size();
    std::vector<cl::Platform> platforms; 
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(!devices.empty());

    cl::Context ctx{devices};
    cl::CommandQueue cmd_queue{ctx, devices[0]};

    cl::Program program;

    try {
        program = cl::Program{ctx, std::string(update_particles_str), false};
        program.build(devices);
    } catch (...) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        std::cerr << build_log << "\n";
    }
    auto kernel = cl::make_kernel<int, cl::Buffer, float>(program, "update_particles");
    const size_t num_workitems = num_particles + block_size - (num_particles % block_size);
    cl::NDRange global_range(num_workitems);
    cl::NDRange local_range(block_size);

    cl::Buffer d_particles{ctx, particles.begin(), particles.end(), false};

    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        kernel(cl::EnqueueArgs(cmd_queue, global_range, local_range), num_particles, d_particles, velocity_update);
    }
    cmd_queue.finish();
    auto end_time = high_resolution_clock::now();
    cl::copy(cmd_queue, d_particles, particles.begin(), particles.end());
    cmd_queue.finish();
    auto duration_count = duration_cast<milliseconds>(end_time - start_time).count();
    return duration_count;
}

int main(int argc, char* argv[]) {
    int num_particles = 10000000;
    int num_iterations = 1000;
    int block_size = 256;
    constexpr float velocity_update = 0.1;

    if (argc == 4) {
        num_particles = std::atoi(argv[1]);
        num_iterations = std::atoi(argv[2]);
        block_size = std::atoi(argv[3]);
    }


    std::vector<Particle> particles(num_particles);
    init_particles(particles);
    std::vector<Particle> particles_cpu(particles.begin(), particles.end());

    size_t gpu_ms = run_gpu_sim(num_iterations, block_size, particles, velocity_update);

    //size_t cpu_ms = update_particles_cpu(num_iterations, particles_cpu, velocity_update);

    //compare_results(particles_cpu, particles);

    std::cout << gpu_ms << "\n";

    return 0;
}
