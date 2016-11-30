/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doax1() {
	int v[N];

	v[0] = 0;
	for (int i=1; i<N; i++)
		v[i] = v[i-1] + i;
}
