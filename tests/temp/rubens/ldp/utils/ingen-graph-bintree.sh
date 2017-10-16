#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}"

# Iterate through the depth of the tree
for depth in {4..24..4}; do
	./graph-generator-narytree.py 2 ${depth} > "${outfolder}/${depth}.dat"
done
