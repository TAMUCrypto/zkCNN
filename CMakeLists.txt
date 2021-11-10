cmake_minimum_required(VERSION 3.10)
project(zkCNN)
set(CMAKE_CXX_STANDARD 14)

link_directories(3rd/hyrax-bls12-381)

include_directories(src)
include_directories(3rd)
include_directories(3rd/hyrax-bls12-381/3rd/mcl/include)

add_subdirectory(src)
add_subdirectory(3rd/hyrax-bls12-381)