#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}"

# Iterate through the depth of the tree
for arity in {1..8}; do
	./graph-generator-narytree.py ${arity} 8 > "${outfolder}/${arity}.dat"
done
