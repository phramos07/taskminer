/*
Toy Benchmark #4: The Basic Task

This toy is the basic example of a task in a DOACROSS loop. We should be able to
catch the task in the call to the function "task" within the loop, since it does
not depend on anything previous.

Inlining is required, since our pass is not interprocedural yet.
*/

#include <stdlib.h>
#include <stdio.h>

void task(int *v, int i) {
  v[i] = v[i] & 0x0000FFFF;
  printf("%d\n", v[i]);
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  int *v;
  int *test;
  v = (int *)malloc(20 * sizeof(int));
  test = (int *)malloc(20 * sizeof(int));
  int i, sum, j;

  sum = 0;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    sum += v[i];
    int tmp = test[i] * test[i];
    tmp ^= i;
    sum += tmp;
    // Theoretically, this function here should become a task.
    #pragma omp task depend(inout:v,0)
    task(v, i);
  }

  return sum;
}

