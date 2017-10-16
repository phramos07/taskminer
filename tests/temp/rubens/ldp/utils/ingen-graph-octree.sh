#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}"

# Iterate through the depth of the tree
for depth in {1..8}; do
	./graph-generator-narytree.py 8 ${depth} > "${outfolder}/${depth}.dat"
done
