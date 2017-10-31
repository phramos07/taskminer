#include <omp.h>
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de
 * Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify */
/*  it under the terms of the GNU General Public License as published by */
/*  the Free Software Foundation; either version 2 of the License, or */
/*  (at your option) any later version. */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful, */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*  GNU General Public License for more details. */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License */
/*  along with this program; if not, write to the Free Software */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 * USA            */
/**********************************************************************************************/
#include <stdio.h>
#include "fib.h"
#include "../../include/time_common.h"

unsigned long long int res;

unsigned long long int fib(unsigned long long int n) {
  unsigned long long int x, y;
  if (n < 2)
    return n;

  #pragma omp task shared(x)
  x = fib(n - 1);
  #pragma omp task shared(y)
  y = fib(n - 2);

	#pragma omp taskwait
  return x + y;
}

void fib0(unsigned long long int n) {
  Instance *I = newInstance(100);

  clock_t beg, end;
  unsigned long long int i;
  for (i = 15; i <= n; i += 5) {
    beg = clock();
  	#pragma omp parallel
  	#pragma omp single
  	#pragma omp task untied
    res = fib(i);

    end = clock();
    printf("Fib(%lld) : %lld\n", i, res);
    addNewEntry(I, i, getTimeInSecs(end - beg));
  }
  printf("\n\n");
  writeResultsToOutput(stdout, I);
  freeInstance(I);
}

int main(int argc, char const *argv[]) {
  unsigned long long int n = atoi(argv[1]);
  fib0(n);

  return 0;
}
