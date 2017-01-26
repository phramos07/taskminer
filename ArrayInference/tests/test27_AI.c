#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  int *array;
  array = (int *)malloc(20 * sizeof(int));

  long long int AI1[7];
  AI1[0] = N + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = AI1[1] + 1;
  AI1[3] = AI1[2] / 4;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  AI1[6] = AI1[5] - 0;
  #pragma acc data pcopy(array[0:AI1[6]])
  {
  #pragma acc kernels
  for (int i = 0; i < N; i++) {
    array[i] = i;
  }
}

  return 0;
}

