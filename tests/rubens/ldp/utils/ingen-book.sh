#! /bin/bash


outfolder="$1"
mkdir -p "${outfolder}/80"

for i in 1000 5000 10000 15000 20000 25000; do
	./binary-line-generator.py 10000 $i 8 40 > "${outfolder}/80/${i}.dat"
done                                         
