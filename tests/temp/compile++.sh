#!/bin/bash

INLINE=false;
CLANG=/usr/local/bin/clang++

$CLANG -c -g -emit-llvm $1 -o $1.bc
opt -S -instnamer -mem2reg $1.bc > $1.mem.bc

