#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}/"{20..100..20}

# Iterate through number of edges (density)
for i in {20..100..20}; do

	# Iterate through number of nodes (graph size)
	for j in 256 512 1024 2048 4096; do
		./graph-generator.py ${j} ${i} > "${outfolder}/${i}/${j}.dat"
	done

done
