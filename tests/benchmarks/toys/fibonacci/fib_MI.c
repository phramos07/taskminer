#include <stdio.h>
#include <omp.h>
#include "../../include/time_common.h"

unsigned long long fib(int n)
{
  unsigned long long i, j;
  if (n<2)
    return n;
  else
    {
      #pragma omp task untied firstprivate(i)
      i=fib(n-1);

      #pragma omp task untied firstprivate(j)
      j=fib(n-2);

      #pragma omp taskwait
      return i+j;
    }
}

int main(int argc, char* argv[])
{
  int n = atoi(argv[1]);

  Instance* I = newInstance(100);

  clock_t beg, end;
  int i;
  for (i = 15; i <= n; i += 5)
  {
  	long long unsigned res;
  	beg = clock();
  	#pragma omp parallel
  	#pragma omp single
  	#pragma omp task untied
  	res = fib(i);
  	printf("Fib(%d) : %lld\n", i, res);
		end = clock();
		addNewEntry(I, i, getTimeInSecs(end-beg));  
  }
  printf("\n\n");

  writeResultsToOutput(stdout, I);
  freeInstance(I);

  return 0;
}