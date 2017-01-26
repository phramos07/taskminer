#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main() {
  int a[300];
  unsigned int i = 0, j = 0, k = 0;
#pragma omp parallel
  {
#pragma omp single
    {
      for (i = 0; i < 300; i++) {
#pragma omp task
        { a[i] = i; }
#pragma omp task
        { k += a[i]; }
      }
    }
  }
  return 0;
}
