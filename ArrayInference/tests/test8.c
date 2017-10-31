#include <stdio.h>

int foo1(int *a, int *b, int n) {
  int i, s = 0;
  for (i = 0; i < n; i++) {
    s = s * a[i];
  }

  for (i = 0; i < n; i++) {
    b[i] = a[i] + 3;
    s += a[i];
  }
  return s;
}

void foo2() {
  int i = 0, j = 2;
  while (i < 10) {
    j += i;
    j *= 2;
    i++;
  }
}

void foo3() {
  int i = 0, j = 0, v[10];
  while (i < 10) {
    v[9 - i] = i;
    i++;
  }
}

void foo4() {
  int i = 0, j = 1, *y = &j;
  while (i < 10) {
    *y = i;
    i++;
  }
}

void fernando_evil_1(int *a, int *b, int N) {
  int i;
  for (i = 0; i < N; i++) {
    a[i] = i;
    b[a[i]] = 0;
  }
  for (i = 0; i < N; i++) {
    int j;
    b[i] += 0;
    for (j = 0; j < N; j++) {
      b[a[j]] = i;
    }
  }
}
