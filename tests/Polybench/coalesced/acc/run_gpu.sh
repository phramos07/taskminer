FILES=$(find . -name "*_gpu")

for f in $FILES; do
	echo "---------- Running $f ----------"
	for num in $(seq 5); do
		echo "----- RUN #${num} -----"
		time ./$f
	done
done
