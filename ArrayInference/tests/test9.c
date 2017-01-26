#include <stdio.h>
#include <stdlib.h>

int Tmp;

void foo(int n) {
  int i;
  for (i = 0; i < n; i++) {
    Tmp += i;
    Tmp = i;
  }
}
