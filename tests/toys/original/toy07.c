/*
Description: two function calls within a loop.

Expectation: ANNOTATE EACH FUNCTION CALL AS A TASK.
Reason: There are dependencies, but if each function call is a task, it'll do
good.
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
  	task(&v[i-1], i);    
  }

  return sum;
}
