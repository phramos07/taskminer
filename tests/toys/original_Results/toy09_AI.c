/*

Description: Function call within a loop with some other stuff in it.
Expectation: ANNOTATE IT.
Reason:

*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>

void minimum(int *a, int *b, int *c) {
  int tmp = 0;

  if (*a > *b) {
    tmp = *b;
    *b = *a;
    *a = tmp;
  }

  if (*a > *c) {
    tmp = *c;
    *c = *a;
    *a = tmp;
  }

  if (*b > *c) {
    tmp = *c;
    *c = *b;
    *b = tmp;
  }
}

int main() {
  int N = 10;
  int u[N], v[N], w[N];
  srand(time(NULL));
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < N; i++) {
    u[i] = rand() % N;
    v[i] = rand() % N;
    w[i] = rand() % N;

    int a, b, c;
    for (int k = 0; k < N * N; k++) {
      a = rand() % N;
      b = rand() % N;
      c = rand() % N;
    }

    long long int TM20[1];
    TM20[0] = (0 / 4);
    #pragma omp task depend(inout:u[TM20[0]],v[TM20[0]],w[TM20[0]])
    minimum(&u[a], &v[b], &w[c]);
  }

  printf("Finishing.\n");
  return 0;
}

