/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doall5() {
	int u[N], v[N];

	for (int i=0; i<N; i++)
		v[i] = u[i];
}
