/*
TOY #2
-> GLOBAL ARRAY
Description: Classic Function Call Task. There's a loop with only one function
call in it, and it can be marked as a task. This one has GLOBAL variables.

Expectation: ANNOTATE IT
Reason: SINGLE FUNCTION CALL TASK. DEPENDENCIES ARE EASY SOLVED BY THE RUNTIME.
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 100

// GLOBAL ARRAY
int u[N];

void one_read(int *a, int *b, int *c) {
  *a = 100;
  *b = 100;
  printf("[one_read] a = %d\n c = %d\n", *a, *c);
}

int main() {
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < N; i++) {
    long long int TM2[8];
    TM2[0] = i * 4;
    TM2[1] = (TM2[0] / 4);
    TM2[2] = i + 1;
    TM2[3] = TM2[2] * 4;
    TM2[4] = (TM2[3] / 4);
    TM2[5] = i + 2;
    TM2[6] = TM2[5] * 4;
    TM2[7] = (TM2[6] / 4);
    #pragma omp task depend(inout:u[TM2[1]],u[TM2[4]],u[TM2[7]])
    one_read(&u[i], &u[i + 1], &u[i + 2]);
  }

  return 0;
}
