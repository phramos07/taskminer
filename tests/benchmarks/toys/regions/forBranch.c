#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define SIZE 1000000
#define HOP 100

int main()
{
	int *results = (int*)malloc(sizeof(int) * SIZE);
	int *results_2 = (int*)malloc(sizeof(int) * SIZE);
	int i, a;

	#pragma omp parallel
	#pragma omp single
	for (int j = HOP; j < SIZE; j+=HOP)
	{
		#pragma omp task depend(inout: results[j], results_2[j])
		{
			if (j%2)
			{
				for (int a = 0; a < SIZE; a++)
					results[j] += results_2[j];
			}
			else
			{
				for (int a = 0; a < SIZE; a++)
					results_2[j] += results[j];
			}
		}
	}

	return 0;
}