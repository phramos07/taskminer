#include <stdio.h>
#include <stdlib.h>

int foo(int *a, int s1, int s2, int n, int m) {
  unsigned int i, j;
  long long int AI1[20];
  AI1[0] = s1 * s2;
  AI1[1] = AI1[0] * 4;
  AI1[2] = AI1[1] / 4;
  AI1[3] = (AI1[2] > 0);
  AI1[4] = (AI1[3] ? AI1[2] : 0);
  AI1[5] = n + -1;
  AI1[6] = AI1[5] - s1;
  AI1[7] = s2 * AI1[6];
  AI1[8] = AI1[0] + AI1[7];
  AI1[9] = s1 + AI1[6];
  AI1[10] = m + -1;
  AI1[11] = AI1[10] - s2;
  AI1[12] = AI1[9] * AI1[11];
  AI1[13] = AI1[8] + AI1[12];
  AI1[14] = AI1[13] * 4;
  AI1[15] = AI1[14] + 4;
  AI1[16] = AI1[15] / 4;
  AI1[17] = (AI1[16] > 0);
  AI1[18] = (AI1[17] ? AI1[16] : 0);
  AI1[19] = AI1[18] - AI1[4];
#pragma acc data pcopy(a[AI1[4] : AI1[19]])
  {
#pragma acc kernels if (!RST_AI1)
    long long int AI1[20];
    AI1[0] = s1 * s2;
    AI1[1] = AI1[0] * 4;
    AI1[2] = AI1[1] / 4;
    AI1[3] = (AI1[2] > 0);
    AI1[4] = (AI1[3] ? AI1[2] : 0);
    AI1[5] = n + -1;
    AI1[6] = AI1[5] - s1;
    AI1[7] = s2 * AI1[6];
    AI1[8] = AI1[0] + AI1[7];
    AI1[9] = s1 + AI1[6];
    AI1[10] = m + -1;
    AI1[11] = AI1[10] - s2;
    AI1[12] = AI1[9] * AI1[11];
    AI1[13] = AI1[8] + AI1[12];
    AI1[14] = AI1[13] * 4;
    AI1[15] = AI1[14] + 4;
    AI1[16] = AI1[15] / 4;
    AI1[17] = (AI1[16] > 0);
    AI1[18] = (AI1[17] ? AI1[16] : 0);
    AI1[19] = AI1[18] - AI1[4];
    #pragma acc data pcopy(a[AI1[4]:AI1[19]])
    {
    #pragma acc kernels if(!RST_AI1)
    for (unsigned int i = s1; i < n; i++) {
      for (int j = s2; j < m; j++) {
        a[i * j] = i * j;
      }
    }
  }
  }
}

