BINS=$(find . -name "*_EXTRALARGE_cpu")

for f in $BINS; do
	echo "Running benchmark: $f:"
	for n in $(seq 10); do
		echo "---------- RUN #$n ----------"
		./$f
	done
done
