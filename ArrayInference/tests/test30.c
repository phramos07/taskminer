#include <stdio.h>

void foo(int *a, int *b, int n) {
  int i;
  int c = 0;

  /*pragma omp target data map(tofrom : a[0 : 1000], b[0 : 1000]) if (0)
    {
  pragma omp target if (0)
  pragma omp for*/
  for (i = 0; i < n; i++) {
    a[i] = b[i];
  }

  /*pragma omp target if (0)
  pragma omp for*/
  for (i = 0; i < n; i++) {
    b[i] = 0;
  }
  //}
  /*pragma acc data copy (a[0:1000])
  pragma acc kernels
  pragma acc loop independent
  for (i = 0; i < n; i++) {
          a[i] = 0;
  }*/
}

int main() {
  int a[1500];
  int b[1500];
  int n = 1000;
  foo(a, b, n);
  return 0;
}
