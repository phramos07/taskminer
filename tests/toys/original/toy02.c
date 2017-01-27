/*
TOY #2
-> GLOBAL ARRAY
Description: Classic Function Call Task. There's a loop with only one function
call in it, and it can be marked as a task. This one has GLOBAL variables.

Expectation: ANNOTATE IT
Reason: SINGLE FUNCTION CALL TASK. DEPENDENCIES ARE EASY SOLVED BY THE RUNTIME.
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 100

//GLOBAL ARRAY
int u[N];

void one_read(int* a, int* b, int* c) {
	*a = 100;
	*b = 100;
	printf("[one_read] a = %d\n c = %d\n", *a, *c);
}

int main()
{
	#pragma omp parallel
	#pragma omp single
	{
		for(int i=0; i<N; i++)
		{
			#pragma omp task depend(inout:u[i])
			one_read(&u[i], &u[i+1], &u[i+2]);
		}
	}

	// printf("Finished. %d\n", u[rand() % N]);
	return 0;
}