#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}"

# Iterate through number of nodes (graph size)
for j in {256..2048..256}; do
	./graph-generator-euler.py ${j} > "${outfolder}/${j}.dat"
done
