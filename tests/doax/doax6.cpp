/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		100

void doax6() {
	int v[N];

	for (int i=0; i<N; i++) {
		v[i] = v[N/2] + 1;
	}
}
