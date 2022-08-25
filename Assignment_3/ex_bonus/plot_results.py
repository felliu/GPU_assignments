import matplotlib.pyplot as plt
import numpy as np

import re

def parse_times():
    cpu_regex = "CPU matmul:.*"
    cublas_regex = "GPU cuBLAS matmul:.*"
    global_regex = r"GPU matmul \(global memory\):.*"
    shared_regex = r"GPU matmul \(shared memory\):.*"
    float_regex = r"([0-9]*[.])?[0-9]+"

    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    times = []
    for sz in sizes:
        time_dict = {}
        time_dict["size"] = sz
        filename = "out_" + str(sz) + ".txt"
        with open(filename, "r") as log_file:
            for line in log_file:
                if re.match(cpu_regex, line):
                    m = re.search(float_regex, line)
                    time_ms = float(m.group(0))
                    time_dict["cpu"] = time_ms;
                elif re.match(cublas_regex, line):
                    m = re.search(float_regex, line)
                    time_ms = float(m.group(0))
                    time_dict["cublas"] = time_ms;
                elif re.match(global_regex, line):
                    m = re.search(float_regex, line)
                    time_ms = float(m.group(0))
                    time_dict["global"] = time_ms;
                elif re.match(shared_regex, line):
                    m = re.search(float_regex, line)
                    time_ms = float(m.group(0))
                    time_dict["shared"] = time_ms;
        times.append(time_dict)

    print(times)
    return times

def make_gpu_plots(times):
    cublas_times = [dict["cublas"] for dict in times]
    global_times = [dict["global"] for dict in times]
    shared_times = [dict["shared"] for dict in times]
    cpu_times = [dict["cpu"] for dict in times]

    sizes = [dict["size"] for dict in times]

    x = np.arange(len(cublas_times))

    width = 0.20

    fig, ax = plt.subplots(1,1)
    ax.set_yscale("log")
    ax.bar(x - 1.5 * width, cublas_times, width, label="cublas")
    ax.bar(x - 0.5 * width, shared_times, width, label="shared memory")
    ax.bar(x + 0.5 * width, global_times, width, label="global_memory")
    ax.bar(x + 1.5 * width, cpu_times, width, label="CPU")

    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("Matrix dimension")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("GEMM performance comparison")

    ax.legend()

    plt.savefig("gpu_barplot.pdf")

if __name__ == "__main__":
    plt.style.use("bmh")
    times = parse_times()
    make_gpu_plots(times)

