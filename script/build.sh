#!/bin/bash
cd ..
mkdir -p cmake-build-release
cd cmake-build-release
/usr/bin/cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" ..
cd ..

if [ ! -d "./data" ]
then
    tar -xzvf data.tar.gz
fi
cd script