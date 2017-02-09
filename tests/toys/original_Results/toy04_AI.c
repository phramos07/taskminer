/*
TOY #4
-> GLOBAL ARRAY
Description: Function with no arguments.

Expectation: DON'T ANNOTTATE
Reason: DEPENDENCY BETWEEN ITERATIONS WHEN WRITING TO ARRAY u
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 100

// GLOBAL ARRAY
int u[N];

void func() {
  int a, b, c;
  a = rand() % N;
  b = (rand() + u[30]) % N;
  c = rand() % N;
  u[a] = b;
  u[b] = a;
  u[c] += u[b] | 0x0000FFFF;
}

int main() {
  srand(time(NULL));
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < N; i++) {
    #pragma omp task
    func();
  }

  return u[rand() % N];
}
