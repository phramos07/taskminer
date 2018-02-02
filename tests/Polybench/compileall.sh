export PATH="${PATH}:/home/periclesrafael/openmp4/llvm/install/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/periclesrafael/openmp4/llvm/install/lib"
export C_INCLUDE_PATH="${C_INCLUDE_PATH}:/home/periclesrafael/openmp4/install/include"
export LIBRARY_PATH="/home/periclesrafael/llvm/install/lib"

export PATH="${PATH}:/opt/pgi/linux86-64/2016/bin"

ACC_FILES=$(find . -path "*acc*_*.c")

for f in $ACC_FILES; do
        PREFIX=${f%_*}
	SIZE=${f##*_}
	SIZE=${SIZE%.*}
	echo "Compiling file $(basename $f)"
        BIN=$PREFIX"_"$SIZE"_gpu"
	SEQ_BIN=$PREFIX"_"$SIZE"_cpu"
	pgcc $f -D$SIZE"_DATASET" -acc -o $BIN
	gcc $f -D$SIZE"_DATASET" -lm -O3 -o $SEQ_BIN
	echo "Generating gpu binary: $BIN"
	echo "Generating cpu binary: $SEQ_BIN"
done
