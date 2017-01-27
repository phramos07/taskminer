/*
Description: two function calls within a loop.

Expectation: DON'T ANNOTATE IT.
Reason: The loop is irregular. We shall treat these tasks differently in the future.
*/

#include <stdlib.h>
#include <stdio.h>

int task(int* v, int i)
{
  *v = *v & 0x0000FFFF;
  printf("%d\n", *v);
  return *v;
}

int main(int argc, char** argv) {
  int * v;
  v = (int*) malloc(20 * sizeof(int));
  int i, sum;

  sum = 0;
  for (i = 0; i < 100; i++)
  {
    task(&v[i], i);
    sum += v[i];
  	v[i+1] = task(&v[i-1], i);    
  }

  return sum;
}
