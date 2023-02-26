#!/bin/bash
cd ..
mkdir -p cmake-build-release
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" ..
make
cd ..

if [ ! -d "./data" ]
then
    tar -xzvf data.tar.gz
fi
cd script
