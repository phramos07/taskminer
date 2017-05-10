/*
TOY #3
Description: Function with no arguments.

Expectation: ANNOTATE IT.
Reason: LOOP IS A DOALL.
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 100

int func() {
  int u[N];
  int a, b, c;
  a = rand() % N;
  b = (rand() + u[30]) % N;
  c = rand() % N;
  u[a] = b;
  u[b] = a;
  u[c] += u[b] | 0x0000FFFF;

  return u[a] + u[b] + u[c];
}

int main() {
  int u[N];
  srand(time(NULL));
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < N; i++) {
    #pragma omp task
    u[i] = func();
  }

  return u[rand() % N];
}
