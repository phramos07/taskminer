/**
 * Example of DOALL loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void f1(int* ptr) {
	int a = *ptr;
}

void doall11() {
	int v[N];

	for (int i=0; i<N; i++)
		f1(&v[i]);
}
