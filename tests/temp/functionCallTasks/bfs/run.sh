export OMP_NUM_THREADS=$1

START_TIME=$SECONDS
./par
ELAPSED_TIME=$(($SECONDS - $START_TIME))

echo $ELAPSED_TIME

for i in `seq 1 10`;
	do
	START_TIME=$SECONDS
	./par
	ELAPSED_TIME2=$(($SECONDS - $START_TIME))
	ELAPSED_TIME=$(( ($ELAPSED_TIME + $ELAPSED_TIME2)/2 ))

	echo $ELAPSED_TIME
	done

(echo $ELAPSED_TIME) &> $1_threads.txt