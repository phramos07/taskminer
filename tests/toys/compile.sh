#!/bin/bash

INLINE=false;
CLANG=/usr/local/bin/clang

$CLANG -c -g -I/Users/pedroramos/programs/csmith-2.2.0/runtime -emit-llvm $1 -o $1.bc
if ($INLINE); then
	opt -S -instnamer -mem2reg -loop-simplify -inline $1.bc > $1.inline.bc
else
	opt -S -instnamer -mem2reg -loop-simplify $1.bc > $1.mem.bc
fi