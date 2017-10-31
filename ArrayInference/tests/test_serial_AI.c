#include <stdlib.h>

void  saxpy_serial(int n, float alpha, float *x, float *y) {
  long long int AI1[5];
  AI1[0] = n + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = AI1[1] / 4;
  AI1[3] = (AI1[2] > 0);
  AI1[4] = (AI1[3] ? AI1[2] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !((x + 0 > y + AI1[4])
  || (y + 0 > x + AI1[4]));
  #pragma omp target device (GPU_DEVICE) if(!RST_AI1)
  #pragma omp target data map(x[0:AI1[4]],y[0:AI1[4]]) if(!RST_AI1)
  #pragma omp parallel for if(!RST_AI1)
  for (int i = 0; i < n; i++)
    y[i] = alpha*x[i] + y[i];
}

