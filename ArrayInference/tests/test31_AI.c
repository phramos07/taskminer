#include <stdio.h>

void foo(int *a, int *b, int n) {
  int i;
  int c = 0;

  {
    long long int AI1[6];
    AI1[0] = n + -1;
    AI1[1] = 4 * AI1[0];
    AI1[2] = AI1[1] + 4;
    AI1[3] = AI1[2] / 4;
    AI1[4] = (AI1[3] > 0);
    AI1[5] = (AI1[4] ? AI1[3] : 0);
    char RST_AI1 = 0;
    RST_AI1 |= !((a + 0 > b + AI1[5])
    || (b + 0 > a + AI1[5]));
    #pragma omp target device (GPU_DEVICE) if(!RST_AI1)
    #pragma omp target data map(tofrom: a[0:AI1[5]],b[0:AI1[5]]) if(!RST_AI1)
    {
    #pragma omp parallel for 
    for (i = 0; i < n; i++) {
      a[i] = b[i];
    }

    #pragma omp parallel for 
    for (i = 0; i < n; i++) {
      b[i] = 0;
    }
  }
  }
}

int main() {
  int a[1500];
  int b[1500];
  int n = 1000;
  foo(a, b, n);
  return 0;
}

