/*

Description: Function call whose parameters are SCALAR only.

Expectation: ANNOTATE IT OR NOT, it doesn't matter.
Reason: There are no dependencies. But annotating scalars does not interest us.
Besides, it is a DOALL.
*/

#include <stdlib.h>
#include <stdio.h>

int task(int* v, int i)
{
  v[i] = v[i] & 0x0000FFFF;
  printf("%d\n", v[i]);
  return v[i];
}

int main(int argc, char** argv) {
  int * v;
  v = (int*) malloc(20 * sizeof(int));
  int i, sum;

  sum = 0;
  for (i = 0; i < 100; i++)
  {
    task(v, i);
  }

  return sum;
}
