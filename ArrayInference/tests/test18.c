#include <stdlib.h>

int *a;

int foo (int n) {
  int i, x;
  for (i = 0; i < n; i++) {
    x = a[i];
    a[i] = x * x;
  }
}
