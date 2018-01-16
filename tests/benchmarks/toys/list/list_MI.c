#include <stdlib.h>
#include "../../include/time_common.h"
#include <omp.h>

/* Forward declarations */
typedef struct TYPE_4__ TYPE_1__;

/* Type definitions */
struct TYPE_4__ {
  int counter;
  int ans;
  struct TYPE_4__ *next;
};
typedef TYPE_1__ LIST;

void foo(LIST *n, int S) {
	#pragma omp parallel
	#pragma omp single
  while (S > 1) {
  	int N = n->counter;
    n->ans = 0;
    // Start task here
    #pragma omp task firstprivate(n)
    for (int j = 0; j < N; j++) {
      n->ans += j;
    }
    // finish task here
    n = n->next;
    S--;
  }
}

int main(int argc, char const *argv[]) {
  Instance *I = newInstance(atoi(argv[1]));
  LIST *L = (LIST *)malloc(sizeof(LIST));

  L->counter = atoi(argv[1]);
  clock_t beg, end;
  beg = clock();
  foo(L, L->counter);
  end = clock();
  addNewEntry(I, 0, getTimeInSecs(end - beg));
  writeResultsToOutput(stdout, I);
  freeInstance(I);

  return 0;
}