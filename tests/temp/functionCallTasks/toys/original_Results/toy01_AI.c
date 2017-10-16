/*
TOY #1
Description: Classic Function Call Task. There's a loop with only one function
call in it, and it can be marked as a task.

Expectation: ANNOTATE IT
Reason: SINGLE FUNCTION CALL TASK. DEPENDENCIES ARE EASY SOLVED BY THE RUNTIME.

*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 100

void one_read(int *a, int *b, int *c) {
  *a = 100;
  *b = 100;
  printf("[one_read] a = %d\n c = %d\n", *a, *c);
}

int main() {
  int u[N];
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < N; i++) {
    long long int TM3[8];
    TM3[0] = i * 4;
    TM3[1] = (TM3[0] / 4);
    TM3[2] = i + 1;
    TM3[3] = TM3[2] * 4;
    TM3[4] = (TM3[3] / 4);
    TM3[5] = i + 2;
    TM3[6] = TM3[5] * 4;
    TM3[7] = (TM3[6] / 4);
    #pragma omp task depend(inout:u[TM3[1]],u[TM3[4]],u[TM3[7]])
    one_read(&u[i], &u[i + 1], &u[i + 2]);
  }

  return 0;
}
