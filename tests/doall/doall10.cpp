/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void doall10() {
	int v[N];

	for (int i=0; i<N/2; i++)
		v[N/2 + i] = 0;
}
