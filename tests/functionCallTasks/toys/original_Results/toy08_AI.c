/*
Description: two function calls within a loop.

Expectation: DON'T ANNOTATE IT.
Reason: The loop is irregular. We shall treat these tasks differently in the
future.
*/

#include <stdlib.h>
#include <stdio.h>

int task(int *v, int i) {
  *v = *v & 0x0000FFFF;
  printf("%d\n", *v);
  return *v;
}

int main(int argc, char **argv) {
  int *v;
  v = (int *)malloc(20 * sizeof(int));
  int i, sum;

  sum = 0;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < 100; i++) {
    long long int TM7[2];
    TM7[0] = i * 4;
    TM7[1] = (TM7[0] / 4);
    #pragma omp task depend(inout:v[TM7[1]],i)
    task(&v[i], i);
    sum += v[i];
    long long int TM9[3];
    TM9[0] = i - 1;
    TM9[1] = TM9[0] * 4;
    TM9[2] = (TM9[1] / 4);
    #pragma omp task depend(inout:v[TM9[2]],i)
    v[i + 1] = task(&v[i - 1], i);
  }

  return sum;
}

