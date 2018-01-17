#include <stdlib.h>
#include <stdio.h>

int foo(int* U, int* V, int N, int M) 
{
	int i, j;
	for(i = 0; i < N; i++) 
	{
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