#include <stdlib.h>
#include <stdio.h>
/*
void func(int a, int b, int *n) {
  for (int i = 0; i < b; i += 2) {
    n[i] = n[i] + 1;
  }
}

void funcA(int a, int b, int *n) {
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < a; j++) {
      n[i * a + j] = i + j;
    }
  }
}

void funcB(int a, int b, int *m) {
  int i;
  for (i = 0; i < a; i++) {
    m[i * 4] = i;
  }
}

void funcC(int n, int *m) {
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      m[i * n + j] = i;
}
*/
int main() {
  int pont[43], i;
  #pragma omp target data map(tofrom: pont[0:43])
  {
  #pragma omp target if(!RST_AI1)
  #pragma omp parallel for 
  for (i = 0; i <= 42; i++)
    pont[i] = 0;
  #pragma omp target if(!RST_AI1)
  #pragma omp parallel for 
  for (i = 0; i < 42; i++)
    pont[i] += 100;
  }
  return 0;
}

