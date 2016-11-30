/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doall4() {
	int v[N];

	for (int i=0; i<N; i++)
		v[i] = 2 * v[i];
}
