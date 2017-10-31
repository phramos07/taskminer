#include <stdlib.h>

void foo(int *v, int n, int m, int o) {
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      v[i * o + j] = i;
}
