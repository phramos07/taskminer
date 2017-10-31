#include <stdlib.h>
#include <stdio.h>

int *vectorE;

void foo1(int n) {
  int i;
  for (i = 0; i < n; i++)
    vectorE[i] = i;
  for (i = 0; i < n; i++)
    vectorE[i + 2] = i;
}

int foo2(int n) {
  int i;
  int sum = 0;
  for (i = 0; i < n; i++)
    sum += vectorE[i];
  return sum;
}

void foo(int n) {
  int i;
  for (i = 0; i < n; i++)
    vectorE[i] = i;
}

int main(int argc, char *argv[]) {
  vectorE = (int *)malloc(sizeof(int) * 100);
  foo(100);
  return 0;
}

