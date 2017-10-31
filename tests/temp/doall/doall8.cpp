/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10


int v[N];
int init = 0;

void doall8() {
	for (int i=0; i<N; i++)
		v[i] = init;
}
