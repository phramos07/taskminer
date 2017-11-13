#include <omp.h>
#include <stdlib.h>
#include "../../include/time_common.h"

#define TRIPCOUNT_CUTOFF 1000

/* Forward declarations */
typedef  struct TYPE_4__   TYPE_1__ ;

/* Type definitions */
struct TYPE_4__ {int counter; int ans; struct TYPE_4__* next; } ;
typedef  TYPE_1__ LIST ;

void foo(LIST* n) {
	#pragma omp parallel
	#pragma omp single
  while (n->next) {
    int N = n->counter;
    n->ans = 0;
    #pragma omp task depend(in: n) if(N > TRIPCOUNT_CUTOFF)
    for (int j = 0; j < N; j++) {
      n->ans += j;
    }
    n = n->next;
  }
}

int main(int argc, char const *argv[])
{
	LIST* L = (LIST*) malloc(sizeof(LIST));
	L->counter = 100;
	foo(L);
	return 0;
}