/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

#define N 		1000

void doax8() {
	int v[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	for (int i=0; i<10; i++) {
		v[ v[i] ] = i;
	}
}
