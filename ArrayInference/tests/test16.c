#include<stdlib.h>
#include<stdio.h>

int foo (int i, int *v) {
  int j;
  for (j = 0; j < 100; j++) {
    if (i%2)
      i = v[i];
    else
      i = foo(v[i], v);
  }
  return i;
}

int main(int argc, char *argv[]) {
  int *k;
  foo(argc, k);
  return 0;
}
