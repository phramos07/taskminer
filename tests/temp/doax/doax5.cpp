/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		100

void doax5() {
	int v[N];

	v[0] = 0;
	v[1] = 1;
	for (int i=2, x=0; i<N; i++) {
		if (i % 2)
			x = v[i-1];
		else
			x = v[i-2];

		v[i] = x;
	}
}
