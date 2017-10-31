#include <stdlib.h>
#include <stdio.h>

void foo(int *p, int n, int m) {
  int v[m];

  for (int i = 0; i < n; i++) {
    p[i] = i;
  }

  for (int i = 0; i < m; i++) {
    v[i] = p[i];
  }
}
