#include <stdio.h>

void foo(int *a, int *b, int n) {
  int i;
  int c = 0;

  {
    for (i = 0; i < n; i++) {
      a[i] = b[i];
    }

    for (i = 0; i < n; i++) {
      b[i] = 0;
    }
  }
}

int main() {
  int a[1500];
  int b[1500];
  int n = 1000;
  foo(a, b, n);
  return 0;
}
