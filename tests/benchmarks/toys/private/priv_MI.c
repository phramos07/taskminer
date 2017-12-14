#include <omp.h>
#include <stdlib.h>

int sum_range(int* V, int N, int L, int U, int* A)
{
	int i=0, sum=0;
	#pragma omp parallel
	#pragma omp single
	while (i < N)
	{
		int j = V[i];
		A[i] = 0;
		#pragma omp untied default(shared) firstprivate(i, j)
		for (; j < L; j++)
		{
			A[i] += V[j];
			j++;				
		}
		i++;		
	}
	return sum;
}

int main(int argc, char const *argv[])
{	
	return 0;
}