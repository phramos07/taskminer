#include <omp.h>
#ifndef TIME_COMMON_H
#define TIME_COMMON_H
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Instance Instance;

struct Instance
{
	int i;
	int n;
	int* sizes;
	double* times;	
};

void writeResultsToOutput(FILE* stream, Instance* I)
{
	int i;
	fprintf(stream, "Size\tTime\n");
	for (i = 0; i < I->i; i++)
		fprintf(stream, "%d\t%.4lf\n", I->sizes[i], I->times[i]);
}

Instance* newInstance(int size)
{
	Instance* I = (Instance*) malloc(sizeof(Instance));
	I->sizes = (int*) malloc(sizeof(int) * size);
	I->times = (double*) malloc(sizeof(double) * size);
	I->i = 0;

	return I;
}

void freeInstance(Instance* I)
{
	free(I->sizes);
	free(I->times);
	free(I);
}

void addNewEntry(Instance* I, int size, double time)
{
	I->sizes[I->i] = size;
	I->times[I->i] = time;
	I->i++;
}

double getTimeInSecs(clock_t tt)
{
	return (double) tt/((double)CLOCKS_PER_SEC);
}

#endif
