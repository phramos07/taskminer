BASE_BIN=$(find . -path "*base*acc*_gpu")
COAL_BIN=$(find . -path "*coalesced*acc*_gpu")

LOG="$(pwd)/nvprof.log"

echo "----- Running Polybench Test suite! -----"

echo "----- Base (Non-Coalesced) OpenACC Benchmarks -----"
for f in $BASE_BIN; do
	#extract benchmark name and data set size
	BENCH="$(basename $f)"
	BENCH="${BENCH%_*_*}"
	SIZE="$(basename $f)"
	SIZE=${SIZE%_*}
	SIZE=${SIZE##*_}

	#run profiling and data movement measurement set
	for run in $(seq 3); do
		echo "---------- PROFILING RUN #$run ----------" >>$LOG
		echo -e "\n\n" >>$LOG
		nvprof --log-file part.log ./$f
		cat part.log >> $LOG
	done
done

echo "----- Copy Coalesced OpenACC Benchmarks -----"
for f in $COAL_BIN; do
	#extract benchmark name and data set size
	BENCH="$(basename $f)"
	BENCH="${BENCH%_*_*}"
	SIZE="$(basename $f)"
	SIZE=${SIZE%_*}
	SIZE=${SIZE##*_}

	#run profiling and data movement measurement set
	for run in $(seq 3); do
		echo "---------- PROFILING RUN #$run ----------" >>$LOG
		echo -e "\n\n" >>$LOG
		nvprof --log-file part.log ./$f
		cat part.log >> $LOG
	done
done
