#include <stdlib.h>
#include <stdio.h>

int main() {
  int a[100][100];
  int i, j;
  acc_create((void*) a, 4040);
  acc_copyin((void*) a, 4040);
  #pragma acc kernels
  #pragma acc loop independent
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
      a[i][j] = i * 10 + j;
    }
  }
  acc_copyout_and_keep((void*) a, 4040);
  return 0;
}

