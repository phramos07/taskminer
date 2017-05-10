bash ~/programs/llvm/lib/Transforms/taskminer/tests/regionTasks/compileOmp.sh $1

echo "Serial"
time ./$1_ser
echo "Parallel"
time ./$1_par
