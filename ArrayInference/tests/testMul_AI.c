#include <stdio.h>
#include <stdlib.h>

void foo(int *a, int s1, int s2, int n, int m) {
  int i, j;
  long long int AI1[19];
  AI1[0] = s1 * m;
  AI1[1] = s2 + AI1[0];
  AI1[2] = AI1[1] * 4;
  AI1[3] = AI1[2] / 4;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  AI1[6] = n + -1;
  AI1[7] = AI1[6] - s1;
  AI1[8] = m * AI1[7];
  AI1[9] = AI1[1] + AI1[8];
  AI1[10] = m + -1;
  AI1[11] = AI1[10] - s2;
  AI1[12] = AI1[9] + AI1[11];
  AI1[13] = AI1[12] * 4;
  AI1[14] = AI1[13] + 4;
  AI1[15] = AI1[14] / 4;
  AI1[16] = (AI1[15] > 0);
  AI1[17] = (AI1[16] ? AI1[15] : 0);
  AI1[18] = AI1[17] - AI1[5];
  #pragma acc data pcopy(a[AI1[5]:AI1[18]])
  #pragma acc kernels
  for (i = s1; i < n; i++) {
    for (j = s2; j < m; j++) {
      a[i * m + j] = i * j;
    }
  }
}

/*void foo(int *a, int s1, int n, int m) {
  int i;
  for (i = s1; i < n; i++) {
    a[i * m + i] = i;
  }
}*/

