/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doall3() {
	int v[N];

	// printf has side-effects, will we still consider this as doall?
	for (int i=0; i<N; i++)
		printf("v[%i] = %d\n", i, v[i]);
}
