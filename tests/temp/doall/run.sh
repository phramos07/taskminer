#!/bin/bash

command -v $OBJ_ROOT/bin/clang++ >/dev/null 2>&1 || { echo >&2 "Please set the correct path to clang++ in the script. Aborting."; exit 1; }
command -v $OBJ_ROOT/bin/opt >/dev/null 2>&1 || { echo >&2 "Please set the correct path to opt in the script. Aborting."; exit 1; }

echo "#############################"
echo "Compiling all cpp files:"
for filename in `ls *.cpp`
do
	printf "%s: \n\t" $filename;
	
	printf "Compiling...";
	$OBJ_ROOT/bin/clang++ -emit-llvm -S $filename -o $filename.ll
	printf "done. ";

	printf "Creating graph...";
	$OBJ_ROOT/bin/opt -disable-opt -load $OBJ_ROOT/lib/LLVMTaskMiner.so -mem2reg -TaskMiner $filename.ll >> $filename.out 2>&1
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
