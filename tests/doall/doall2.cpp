/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doall2() {
	int v[N], i=0;

	for (; i<N/2; i++)
		v[i] = 0;

	for (; i<N; i++)
		v[i] = 0;
}
