#!/bin/bash

INLINE=false;
CLANG=/usr/local/bin/clang
ICSMITH=-I/Users/pedroramos/programs/csmith-2.2.0/runtime 
IBOTS=-I/Users/pedroramos/programs/llvm/lib/Transforms/taskminer/tests/bots/common

$CLANG -c -g $IBOTS -emit-llvm $1 -o $1.bc
opt -S -instnamer -mem2reg -loop-simplify $1.bc > $1.mem.bc

