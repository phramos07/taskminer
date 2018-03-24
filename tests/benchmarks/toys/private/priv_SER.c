#include <omp.h>
#include <stdlib.h>

int sum_range(int* V, int N, int L, int* A)
{
	int i=0, sum=0;
	while (i < N)
	{
		int j = V[i];
		A[i] = 0;
		for (; j < L; j++)
		{
			A[i] += V[j];
			j++;				
		}
		i++;		
	}
	return sum;
}

int main(int argc, char const *argv[]) {
	int n = atoi(argv[1]);
	int *V = malloc(sizeof(int)*n);
	int *A = malloc(sizeof(int)*n);
	sum_range(V, atoi(argv[1]), atoi(argv[1])/10, A); 
	return 0; 
}
