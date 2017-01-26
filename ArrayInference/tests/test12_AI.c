#include <stdlib.h>

void foo(int *v, int n, int m, int o) {
  int i, j;
  #pragma acc kernels
  for (i = 0; i < n; i++)
    long long int AI1[10];
    AI1[0] = n + -1;
    AI1[1] = o * AI1[0];
    AI1[2] = m + -1;
    AI1[3] = AI1[1] + AI1[2];
    AI1[4] = AI1[3] * 4;
    AI1[5] = AI1[4] + 4;
    AI1[6] = AI1[5] / 4;
    AI1[7] = (AI1[6] > 0);
    AI1[8] = (AI1[7] ? AI1[6] : 0);
    AI1[9] = AI1[8] - 0;
    #pragma acc data pcopy(v[0:AI1[9]])
    {
    for (j = 0; j < m; j++)
      v[i * o + j] = i;
}
}

