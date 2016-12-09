/**
 * Example of DOALL loop that requires interprocedural analysis.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		10

void fun1(int* p1, int* p2, int* p3) {
	*p3 = *p1 + *p2;
}

void interp1() {
	int u[N], v[N];

	for (int i=1; i<N; i++) {
		fun1(&u[i-1], &u[i], &v[i]);
	}
}
