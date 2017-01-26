#include <stdlib.h>
#include <stdio.h>

int main() {
  int a[100][100];
  int i, j;
  #pragma acc data copy(a)
  #pragma acc kernels
  #pragma acc loop independent
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      a[i][j] = i * 10 + j;
    }
  }
  return 0;
}
