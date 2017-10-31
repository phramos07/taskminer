#include <stdlib.h>
#include <stdio.h>

void foo(int *p, int n, int m) {
  int v[m];

  long long int AI1[14];
  AI1[0] = m + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = n + -1;
  AI1[3] = 4 * AI1[2];
  AI1[4] = AI1[1] > AI1[3];
  AI1[5] = (AI1[4] ? AI1[1] : AI1[3]);
  AI1[6] = AI1[5] + 4;
  AI1[7] = AI1[6] / 4;
  AI1[8] = (AI1[7] > 0);
  AI1[9] = (AI1[8] ? AI1[7] : 0);
  AI1[10] = AI1[1] + 4;
  AI1[11] = AI1[10] / 4;
  AI1[12] = (AI1[11] > 0);
  AI1[13] = (AI1[12] ? AI1[11] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !((p + 0 > v + AI1[13])
  || (v + 0 > p + AI1[9]));
  #pragma acc data pcopy(p[0:AI1[9]],v[0:AI1[13]]) if(!RST_AI1)
  {
  #pragma acc kernels
  for (int i = 0; i < n; i++) {
    p[i] = i;
  }

  #pragma acc kernels
  for (int i = 0; i < m; i++) {
    v[i] = p[i];
  }
}
}

