#!/bin/bash

for sz in 64 128 256 512 1024 2048 4096
do
    ./ex3 -s $sz -v > out_${sz}.txt
done




