/*
Description: two function calls within a loop.

Expectation: ANNOTATE EACH FUNCTION CALL AS A TASK.
Reason: There are dependencies, but if each function call is a task, it'll do
good.
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
    long long int TM8[3];
    TM8[0] = i - 1;
    TM8[1] = TM8[0] * 4;
    TM8[2] = (TM8[1] / 4);
    #pragma omp task depend(inout:v[TM8[2]],i)
    task(&v[i - 1], i);
  }

  return sum;
}

