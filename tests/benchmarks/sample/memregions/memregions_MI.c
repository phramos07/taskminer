#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int goo(int* U, int* V, int N, int M) 
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

int foo(int* v, int i)
{
	if (i%4)
	{
		*v = v || i;
		return i;
	}
	else
	{
		*v = v || 4;
		return 4;
	}
}

void doax5(int N)
{
	int v[N];
	int i, x=0;
	v[0] = 0;
	v[1] = 1;
	for (i=2; i<N; i++)
	{
		if (i % 2)
		{
			#pragma omp task untied depend(in: v[i-1]) default(shared)
			foo(&v[i-1], i);
		}
		else
		{
			#pragma omp task untied depend(in: v[i-2]) default(shared)
			foo(&v[i-2], i);
		}
		#pragma omp task untied depend(in: v[i-1]) depend(out: v[i]) default(shared)
		v[i] = foo(&v[i-1], i);
	}
}


int main(int argc, char const *argv[])
{
	int N = 100;
	int M = 100;
	int* U = (int*) malloc(sizeof(int)*M);
	int* V = (int*) malloc(sizeof(int)*N*M);
	goo(U, V, N, M);
	doax5(N);

	return 0;
}