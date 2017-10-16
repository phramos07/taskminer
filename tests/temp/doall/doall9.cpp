/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doall9() {
	int v[N];

	for (int i=0; i<N; i++) {
		v[i] = 0;

		for (int j=0; j<N; j++) {
			v[i] = 0;
		}
	}
}
