#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int foo(int* U, int* V, int N, int M) 
{
	int i, j;
	#pragma omp parallel
	#pragma omp single
	for(i = 0; i < N; i++) 
	{
		#pragma omp task depend(in: V[i*N:i*N+M])
		for (j = 0; j < M; j++) 
		{
			int* p = &V[i*N + j];
			U[i] += *p;
		}
	}

	return 1;
}

int main(int argc, char const *argv[])
{
	int N = 100;
	int M = 100;
	int* U = (int*) malloc(sizeof(int)*M);
	int* V = (int*) malloc(sizeof(int)*N*M);
	foo(U, V, N, M);

	return 0;
}