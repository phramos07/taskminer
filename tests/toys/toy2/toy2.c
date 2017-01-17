/*
Toy Benchmark #2: Tasks Regions

There are 2 loops here, each loop has a candidate task region.

The first one should have INS{ } and OUTS{u[i]}, the task region comprising of 
the write access to u[i].

The second one should have INS{u[i]} and OUTS{u[i]}, since it's being stored
and loaded.

Inlining of the function one_read is required, because our TaskMiner is not 
interprocedural.
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>

void one_read(int* a, int* b, int* c) {
	*a = 100;
	*b = 100;
	printf("[one_read] a = %d\n c = %d\n", *a, *c);
}

int main() {
	const int N = 100000;
	int u[N];

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