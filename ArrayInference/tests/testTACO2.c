#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void corr(float *A, float *MEAN, float *STDEV, int m, int n) {
  long long int AI1[14];
  AI1[0] = m + -1;
  AI1[1] = n * AI1[0];
  AI1[2] = n + -1;
  AI1[3] = AI1[1] + AI1[2];
  AI1[4] = AI1[3] * 4;
  AI1[5] = AI1[4] + 4;
  AI1[6] = AI1[5] / 4;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  AI1[9] = 4 * AI1[0];
  AI1[10] = AI1[9] + 4;
  AI1[11] = AI1[10] / 4;
  AI1[12] = (AI1[11] > 0);
  AI1[13] = (AI1[12] ? AI1[11] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > MEAN + AI1[13])
  || (MEAN + 0 > A + AI1[8]));
  RST_AI1 |= !((A + 0 > STDEV + AI1[13])
  || (STDEV + 0 > A + AI1[8]));
  RST_AI1 |= !((MEAN + 0 > STDEV + AI1[13])
  || (STDEV + 0 > MEAN + AI1[13]));
  #pragma omp target device (GPU_DEVICE) if(!RST_AI1)
  #pragma omp target data map(to: A[0:AI1[8]]) map(tofrom: MEAN[0:AI1[13]],STDEV[0:AI1[13]]) if(!RST_AI1)
  {
  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      MEAN[i] += A[i * n + j];

    MEAN[i] /= n;
  }

  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      STDEV[i] += (A[i * n + j] - MEAN[i]) * (A[i * n + j] - MEAN[i]);

    STDEV[i] = sqrt(STDEV[i] / n);
  }
}
}

int main(int argc, char *argv[]) {
  float *x, *y, *z, m, n;
  n = 10000;
  m = 10000;
  x = (float *)malloc(sizeof(float) * 100000000);
  y = (float *)malloc(sizeof(float) * 100000000);
  z = (float *)malloc(sizeof(float) * 100000000);
  char RST_AI1 = 0;
  RST_AI1 |= !((x + 0 > y + 99999999)
  || (y + 0 > x + 99999999));
  RST_AI1 |= !((x + 0 > z + 99999999)
  || (z + 0 > x + 99999999));
  RST_AI1 |= !((y + 0 > z + 99999999)
  || (z + 0 > y + 99999999));
  #pragma omp target device (GPU_DEVICE) if(!RST_AI1)
  #pragma omp target data map(tofrom: x[0:99999999],y[0:99999999],z[0:99999999]) if(!RST_AI1)
  {
  #pragma omp parallel for
  for (unsigned int i = 0; i < 100000000; i++) {
    x[i] = i;
    y[i] = i;
    z[i] = i;
  }
  }
  corr(x, y, z, m, n);
  free(x);
  free(y);
  free(z);
}

