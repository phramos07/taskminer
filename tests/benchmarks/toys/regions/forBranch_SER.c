#include <stdlib.h>
#include <math.h>
#define SIZE 1000000
#define HOP 100

int main()
{
	int *results = (int*)malloc(sizeof(int) * SIZE);
	int *results_2 = (int*)malloc(sizeof(int) * SIZE);
	int i, a;
	for (int j = HOP; j < SIZE; j+=HOP)
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

	return 0;
}