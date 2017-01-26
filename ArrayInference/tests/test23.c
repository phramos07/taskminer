#include <stdlib.h>

void foo(int n) {

  int a = 0, b = 0, c = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      a += c + b + i + j;
  c = a + b;

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      a = c + b;
  c = a + b;

  printf("%d", a);
}
