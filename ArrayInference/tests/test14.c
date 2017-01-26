#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  int *v = (int*) malloc(sizeof(int) * n);
  int *vt = (int*) malloc(sizeof(int) * n);
  int i, k = 0;
  for (i = 0; i < n; i++) {
    if (k == 100)
      v[k] = vt[i];
    else
      v[(i-1)] += n;
    k += i;
  }
  return 0;
}
