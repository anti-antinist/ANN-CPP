cmake_minimum_required(VERSION 3.22.1) 
project(neuralnet)

set(SOURCES
    src/main.cpp
)

add_executable(neuralnet ${SOURCES})
target_include_directories(neuralnet PRIVATE include)
