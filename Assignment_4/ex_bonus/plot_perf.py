import matplotlib.pyplot as plt
import numpy as np

def plot_gpu_block_sz():
    fig, ax = plt.subplots(1,1)
    block_sizes = [16, 32, 64, 128, 256]
    num_particles = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000]
    num_particles_str = ["1e4", "2e4", "4e4", "8e4", "1.6e5", "3.2e5", "6.4e5", "1.28e6"]
    x = np.arange(len(num_particles))
    width = 0.13
    offsets = [-2, -1, 0, 1, 2]
    for (i, block_sz) in enumerate(block_sizes):
        with open("gpu_times_" + str(block_sz) + ".txt", "r") as f:
            times = list(map(int, f.read().splitlines()))
            times = [t / 1000 for t in times]
            ax.bar(x + offsets[i] * width, times, width, edgecolor="black", label = str(block_sz))

    ax.legend()
    ax.set_title("GPU block size comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(num_particles_str)
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Execution time (s)")
    plt.show()

def plot_cpu_gpu():
    fig, ax = plt.subplots(1,1)
    num_particles = [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000]
    num_particles_str = ["1e4", "2e4", "4e4", "8e4", "1.6e5", "3.2e5", "6.4e5", "1.28e6"]
    x = np.arange(len(num_particles))
    width = 0.3

    gpu_times = []
    cpu_times = []

    with open("gpu_times_32.txt", "r") as f:
        gpu_times = list(map(int, f.read().splitlines()))
        gpu_times = [t / 1000 for t in gpu_times]

    with open("cpu_times.txt", "r") as f:
        cpu_times = list(map(int, f.read().splitlines()))
        cpu_times = [t / 1000 for t in cpu_times]

    ax.bar(x - 0.5 * width, gpu_times, width, edgecolor="black", label="GPU")
    ax.bar(x + 0.5 * width, cpu_times, width, edgecolor="black", label="CPU")

    ax.legend()
    ax.set_title("GPU/CPU comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(num_particles_str)
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Execution time (s)")
    plt.show()


if __name__ == "__main__":
    plt.style.use("bmh")
    #plot_gpu_block_sz()
    plot_cpu_gpu()

