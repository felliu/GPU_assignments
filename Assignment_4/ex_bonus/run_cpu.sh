#!/bin/bash

. /opt/spack/share/spack/setup-env.sh
spack load cuda@11.2.0
nvcc -O3 -arch=sm_61 main.cpp -lOpenCL -o bonus

touch cpu_times.txt
for n_particles in 10000 20000 40000 80000 160000 320000 640000 1280000
do
    ./bonus ${n_particles} 20000 256 >> cpu_times.txt
done


