#include <stdio.h>
#include <stdlib.h>

void foo(int *a, int s1, int s2, int n, int m) {
  int i, j;
  for (i = s1; i < n; i++) {
    for (j = s2; j < m; j++) {
      a[i * m + j] = i * j;
    }
  }
}

/*void foo(int *a, int s1, int n, int m) {
  int i;
  for (i = s1; i < n; i++) {
    a[i * m + i] = i;
  }
}*/
