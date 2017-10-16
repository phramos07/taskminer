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
