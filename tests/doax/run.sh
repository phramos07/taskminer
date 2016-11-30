#!/bin/bash

for filename in `ls *.cpp`
do
	printf "%s: \n\t" $filename;
	
	printf "Compiling...";
	$OBJ_ROOT/bin/clang++ -emit-llvm -S $filename -o $filename.ll
	printf "done. ";

	printf "Creating graph...";
	$OBJ_ROOT/bin/opt -disable-opt -load LLVMTaskFinder.so -mem2reg -TaskFinder $filename.ll >> $filename.out 2>&1
	printf "done.\n";
done

echo "#############################"
echo "Now creating image files:"
for filename in `ls *.dot`
do
	printf "%s: \n\t" $filename;

	printf "Creating image...";
	dot -T png $filename -o $filename.png
	printf "done.\n";
done
