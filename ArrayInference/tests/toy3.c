/*
Toy Benchmark #3: Catching tasks in irregular loops

This program has loops with iteration-dependencies that make them irregular.
One of the many goals in our analysis is to be able to catch these kind of
dependencies.

*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// void minimum(int* a, int* b, int* c) {
// 	int tmp = 0;

// 	if (*a > *b) {
// 		tmp = *b;
// 		*b = *a;
// 		*a = tmp;
// 	}

// 	if (*a > *c) {
// 		tmp = *c;
// 		*c = *a;
// 		*a = tmp;
// 	}

// 	if (*b > *c) {
// 		tmp = *c;
// 		*c = *b;
// 		*b = tmp;
// 	}
// }

int main() {
  int N = 10;
  int u[N], v[N], w[N];

  // for (int x = 0; x < N; x += 1) {
  // 	u[x] = 0;
  // 	u[x+1] = 1;
  // }

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
#pragma omp task depend(in : u[i], v[i], w[i])
    {
      // These next 2 lines show an iteration dependency that our pass does not
      // catch yet.
      u[i] = rand() % N;
      // u[0] += u[i];
      v[i] = rand() % N;
      w[i] = rand() % N;

      int a, b, c;
      // for(int k=0; k<N*N; k++)
      // {
      // 	a = rand() % N;
      // 	b = rand() % N;
      // 	c = rand() % N;
      // }

      // minimum(&u[a], &v[b], &w[c]);
    }
  }

  printf("Finishing.\n");
  return 0;
}
