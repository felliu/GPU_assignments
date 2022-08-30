#!/bin/bash

. /opt/spack/share/spack/setup-env.sh
spack load cuda@11.2.0
nvcc -O3 -arch=sm_61 main.cpp -lOpenCL -o bonus

for block_sz in 16 32 64 128 256
do
touch gpu_times_${block_sz}.txt
    for n_particles in 10000 20000 40000 80000 160000 320000 640000 1280000
    do
        ./bonus ${n_particles} 20000 $block_sz >> gpu_times_${block_sz}.txt
    done
done


