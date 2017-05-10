#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define SIZE 1000000
#define HOP 1000

int main()
{
	int *results = (int*)malloc(sizeof(int) * SIZE);
	int *results_2 = (int*)malloc(sizeof(int) * SIZE);
	int i, a;

	#pragma omp parallel
	#pragma omp single
	for (int j = HOP; j < SIZE; j+=HOP)
	{
		#pragma omp task depend(inout: results[j], results[j-1], results_2[j])
		{
			a = SIZE-1;
			while (a > 1)
			{
				results[j] += results[j-1] ^ 0x0000FFFF;
				a--;
			}

			for (int j = 0; j < SIZE; j++)
				results_2[a] += results[a];

			results_2[j] += results_2[j] + j + results[j];
		}
	}

	return 0;
}