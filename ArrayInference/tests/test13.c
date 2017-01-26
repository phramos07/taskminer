#include <stdlib.h>
#include <stdio.h>

int foo (int *V) {
  int i;
  for (i = 0; i < 100; i++)
    V[i] = i;
}
