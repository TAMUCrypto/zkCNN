#!/bin/bash

set -x

./build.sh
/usr/bin/cmake --build ../cmake-build-release --target demo_lenet_run -- -j 6

run_file=../cmake-build-release/src/demo_lenet_run
out_file=../output/single/demo-result-lenet5.txt

mkdir -p ../output/single
mkdir -p ../log/single

lenet_i=../data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-images-weights-qint8.csv
lenet_c=../data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv
lenet_o=../output/single/lenet5.mnist.relu.max-1-infer.csv

${run_file} ${lenet_i} ${lenet_c} ${lenet_o} 1 > ${out_file}