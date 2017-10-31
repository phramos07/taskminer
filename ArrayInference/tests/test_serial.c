#include <stdlib.h>

void  saxpy_serial(int n, float alpha, float *x, float *y) {
  for (int i = 0; i < n; i++)
    y[i] = alpha*x[i] + y[i];
}
