/*
TOY #1
Description: Classic Function Call Task. There's a loop with only one function
call in it, and it can be marked as a task.

Expectation: ANNOTATE IT
Reason: SINGLE FUNCTION CALL TASK. DEPENDENCIES ARE EASY SOLVED BY THE RUNTIME.

*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 100

void one_read(int* a, int* b, int* c) {
	*a = 100;
	*b = 100;
	printf("[one_read] a = %d\n c = %d\n", *a, *c);
}

int main() {
	int u[N];
	for(int i=0; i<N; i++)
	{
		one_read(&u[i], &u[i+1], &u[i+2]);
	}

	return 0;
}