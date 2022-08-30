import matplotlib.pyplot as plt
import numpy as np

def make_plot():
    sizes = [10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    gpu_times = [0.066191, 0.066318, 0.069341, 0.079067, 0.202933, 1.460826]

    fig, ax = plt.subplots(1,1)
    ax.loglog(sizes, gpu_times, label="time", marker="*", linestyle="none")

    ax.set_xlabel("Array size")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("OpenACC saxpy")

    plt.show()

if __name__ == "__main__":
    plt.style.use("bmh")
    make_plot()

