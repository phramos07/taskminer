#include <stdlib.h>

int foo(int tmp, int x) {
  return (tmp * x);
}

int main() {
  int v[1000];
  int i, ret = 0;
  for (i = 0; i < 100; i++) {
    if (foo(*(&v + i),i))
    //if (foo(v[i],i))
      ret = 1;
    else
      ret = 0;
  }

  return 0;
}
