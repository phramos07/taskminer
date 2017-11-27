all: example
 
exemple: toy2.c
    LIBRARY_PATH=$(shell dirname $(shell which clang-3.8))/../lib clang-3.8
-fopenmp -omptargets=nvptx64sm_30-nvidia-linux -g -std=c99 $< -o $@
 
