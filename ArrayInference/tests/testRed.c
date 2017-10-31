#include <stdlib.h>
#include <stdio.h>

int main() {
  int i,j,sum1 = 0, sub1 = 0;
  int v[10];
  
  for (i = 0; i < 10; i++)
    v[i] = i;

  for (i = 0; i < 10; i++)
    sub1 -= v[i];

  int sub2 = 0, sum2 = 0;

  for (i = 9; i > -1; i--)
    sub2 -= v[i];

  for (i = 9; i > -1; i--)
    sum2 += (-1 * v[i]);

  for (i = 0; i < 10; i++)
    sum1 += (-1 * v[i]);

  printf("\n%d =?= %d\n", sub1, sum1);
  printf("\n%d =?= %d\n", sub2, sum2);
  return 0;
}
