/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		100

void doax2() {
	int v[N];

	v[0] = 0;
	for (int i=1; i<N; i++) {
		v[i + 1] = v[i] + 1;
	}
}
