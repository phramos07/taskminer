#!bin/bash

for (( i = 0; i < 100; i++ )); do
	csmith > $i.c
	#statements
done