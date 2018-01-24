/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		100

int foo(int* v);

int goo(int* v);

void doax5() {
	int v[N];

	v[0] = 0;
	v[1] = 1;
	for (int i=2, x=0; i<N; i++)
	{
		if (i % 2)
		{
			#pragma omp task depend(in:v[i-1]) default(shared)
			x = foo(&v[i-1]);
		}
		else
		{
			#pragma omp task depend(in:v[i-2]) default(shared)
			x = foo(&v[i-2]);
		}

		#pragma omp task depend(in:v[i-2]) depend(out:v[i]) default(shared)
		v[i] = goo(&v[i-1]);
	}
}
