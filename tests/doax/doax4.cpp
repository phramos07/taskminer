/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		100

void doax4() {
	int v[N];

	v[0] = 0;
	for (int i=1, x=0; i<N; i++) {
		if (i % 2)
			x = v[i-1];
		else
			x = i;

		v[i] = x;
	}
}
