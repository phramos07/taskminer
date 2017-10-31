#include <stdio.h>
#include <stdlib.h>

int **v;

void foo(int n) {
  int i;
  for (i = 0; i < n; i++)
    *v[i] += i;
}

