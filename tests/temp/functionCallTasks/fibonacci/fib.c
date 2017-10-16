#include <stdio.h>
#include <omp.h>

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

int main()
{
  int n = 37;
  {
    printf ("fib(%d) = %lld\n", n, fib(n));
  }

  return 0;
}
