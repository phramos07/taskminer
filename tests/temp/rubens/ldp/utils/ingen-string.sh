#! /bin/bash


StringGenerator() {
	tr -dc '[:alnum:] ,.!?;' < /dev/urandom | tr -d \''\\'\` | head -c $1
}

outfolder="$1"
mkdir -p "${outfolder}/FixedText"
mkdir -p "${outfolder}/FixedPattern"

for i in 1 4 16 64 256 1024; do

	# Fixed length pattern (1 MB)
	outfile="${outfolder}/FixedPattern/$i.dat"

	StringGenerator 1K | tr '\n' '-' > ${outfile}
	echo >> ${outfile}
	StringGenerator ${i}M >> ${outfile}

done

for i in 32 64 128 256 512 1024; do

	# Fixed length text (512 MB)
	outfile="${outfolder}/FixedText/$i.dat"

	StringGenerator ${i} | tr '\n' '-' > ${outfile}
	echo >> ${outfile}
	StringGenerator 128M >> ${outfile}

done
