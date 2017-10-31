#include <stdio.h>
#include "../../include/time_common.h"

unsigned long long int fib(int n)
{
  unsigned long long int i, j;
  if (n<2)
    return n;
  else
    {
      i=fib(n-1);
      j=fib(n-2);
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
  	beg = clock();
  	printf("Fib(%d) : %lld\n", i, fib(i));
		end = clock();
		addNewEntry(I, i, getTimeInSecs(end - beg));  
  }
  printf("\n\n");

	writeResultsToOutput(stdout, I);
  freeInstance(I);

}