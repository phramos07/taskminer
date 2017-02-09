/*

Description: FUNCTION CALL WITHIN AN IRREGULAR LOOP.

Expectation: DON'T ANNOTATE IT.
Reason: THE LOOP IS IRREGULAR AND THERE'S A DEPENDENCY BETWEEN LINES 27 AND 28.
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
    sum += v[i - 1];
    long long int TM8[2];
    TM8[0] = i * 4;
    TM8[1] = (TM8[0] / 4);
    #pragma omp task depend(inout:v[TM8[1]],i)
    task(&v[i], i);
  }

  return sum;
}

