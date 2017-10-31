#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}"

# Iterate through number of nodes (graph size)
for j in {256..2048..256}; do
	./graph-generator-barabasi.py ${j} 16 > "${outfolder}/${j}.dat"
done
