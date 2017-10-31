#include <stdlib.h>
#include "../../include/time_common.h"

/* Forward declarations */
typedef  struct TYPE_4__   TYPE_1__ ;

/* Type definitions */
struct TYPE_4__ {int counter; int ans; struct TYPE_4__* next; } ;
typedef  TYPE_1__ LIST ;

void foo(LIST* n) {
  while (n->next) {
    int N = n->counter;
    n->ans = 0;
    // Start task here
    for (int j = 0; j < N; j++) {
      n->ans += j;
    }
    // finish task here
    n = n->next;
  }
}

int main(int argc, char const *argv[])
{
	Instance* I = newInstance(atoi(argv[1]));
	LIST* L = (LIST*) malloc(sizeof(LIST));

	L->counter = atoi(argv[1]);
	clock_t beg, end;
	beg = clock();
	foo(L);
	end = clock();
	addNewEntry(I, 0, getTimeInSecs(end - beg));
	writeResultsToOutput(stdout, I);
  freeInstance(I);

	return 0;
}