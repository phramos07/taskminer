#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <alloca.h>
#define DEPTH_CUTOFF 10
int taskminer_depth_cutoff;

int ok(int n, char *a) { return 1; }

void nqueens(int n, int j, char *a, int *solutions)
{
	taskminer_depth_cutoff++;

	int *csols;
	int i;

	if (n == j)
	{
		*solutions = 1;
		return;
	}

	*solutions = 0;
	csols = alloca(n*sizeof(int));
	memset(csols,0,n*sizeof(int));

	for (i = 0; i < n; i++)
	{
		char * b = alloca(n * sizeof(char));
		memcpy(b, a, j * sizeof(char));
		b[j] = (char) i;
		if (ok(j + 1, b))
		{
			#pragma omp task untied depend(in: b) depend(inout: solutions) if(taskminer_depth_cutoff < DEPTH_CUTOFF)
			nqueens(n, j + 1, b,&csols[i]);
		}
	}

	#pragma omp taskwait
	for ( i = 0; i < n; i++)
	{
		*solutions += csols[i];
	}

}

int main(int argc, char const *argv[])
{
	int size = 13;
	int total_count = 0;
	char *a;
	a = alloca(size * sizeof(char));

	taskminer_depth_cutoff = 0;
	#pragma omp parallel
	#pragma omp single
	#pragma omp task untied
	nqueens(size, 0, a, &total_count);

	return 0;
}

