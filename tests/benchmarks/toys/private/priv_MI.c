#include <omp.h>
#include <stdlib.h>

int sum_range(int* V, int N, int L, int U, int* A)
{
	int i=0, sum=0;
	#pragma omp parallel
	#pragma omp single
	for (i = 0; i < N; i++)
	{
		#pragma omp untied default(shared) firstprivate(i)
		{
			int j = V[i];
			A[i] = 0;
			for (; j < L; j++)
			{
				A[i] += V[j];
				j++;				
			}
			sum += A[i];
		}
	}

	return sum;
}

int main(int argc, char const *argv[])
{	
	return 0;
}