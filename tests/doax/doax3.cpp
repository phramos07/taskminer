/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		100

void doax3() {
	int v[N];

	v[0] = 0;
	v[1] = 1;
	for (int i=2; i<N; i++) {
		v[i] = v[i-1] + v[i-2];
	}
}
