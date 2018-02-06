#!/bin/bash


sed -i '/device(GPU_DEVICE)/d' $1
sed -i '/pragma omp/d' $1 
sed -i '/pragma target/d' $1 
