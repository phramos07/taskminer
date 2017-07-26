#include <stdio.h>
#include <omp.h>

unsigned long long int fib(int n)
{
  unsigned long long int i, j;
  if (n<2)
    return n;
  else
    {
      #pragma omp task depend(in:n-1) depend(out:i)
      i=fib(n-1);

      #pragma omp task depend(in:n-2) depend(out:j)
      j=fib(n-2);

      #pragma omp taskwait
      return i+j;
    }
}

int main()
{
  int n = 37;

  omp_set_num_threads(8);

  #pragma omp parallel shared(n)
  {
    #pragma omp single
    printf ("fib(%d) = %lld\n", n, fib(n));
  }

  return 0;
}