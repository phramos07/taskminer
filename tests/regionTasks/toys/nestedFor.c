#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define SIZE 1000000
#define HOP 1000

int main()
{
	int *results = (int*)malloc(sizeof(int) * SIZE);
	int *results_2 = (int*)malloc(sizeof(int) * SIZE);

	#pragma omp parallel
	#pragma omp single
	for (int a = HOP; a < SIZE; a+=HOP)
	{
		#pragma omp task depend(inout: results[a], results_2[a])
		{
			results[a] = 0;
			for (int j=1; j < SIZE; j++)
			{
				results[a] += j;
				results[a] = results[a] || 0x0000FFFF + j;
				// results[a] += results[a-HOP] && 0x0000FFFF + j;
			}

			for (int j = 0; j < SIZE; j++)
				results_2[a] += results[a];
		}
	}

	return 0;
}