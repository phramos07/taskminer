#include <stdio.h>

int foo1(int *a, int *b, int n) {
  int i, s = 0;
  long long int AI1[6];
  AI1[0] = n + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = AI1[1] + 4;
  AI1[3] = AI1[2] / 4;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  #pragma acc data pcopyin(a[0:AI1[5]]) 
  #pragma acc kernels
  for (i = 0; i < n; i++) {
    s = s * a[i];
  }

  long long int AI2[6];
  AI2[0] = n + -1;
  AI2[1] = 4 * AI2[0];
  AI2[2] = AI2[1] + 4;
  AI2[3] = AI2[2] / 4;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  char RST_AI2 = 0;
  RST_AI2 |= !((a + 0 > b + AI2[5])
  || (b + 0 > a + AI2[5]));
  #pragma acc data pcopyin(a[0:AI2[5]]) pcopy(b[0:AI2[5]]) if(!RST_AI2)
  #pragma acc kernels if(!RST_AI2)
  for (i = 0; i < n; i++) {
    b[i] = a[i] + 3;
    s += a[i];
  }
  return s;
}

void foo2() {
  int i = 0, j = 2;
  #pragma acc data 
  #pragma acc kernels
  while (i < 10) {
    j += i;
    j *= 2;
    i++;
  }
}

void foo3() {
  int i = 0, j = 0, v[10];
  while (i < 10) {
    v[9 - i] = i;
    i++;
  }
}

void foo4() {
  int i = 0, j = 1, *y = &j;
  #pragma acc data 
  #pragma acc kernels
  while (i < 10) {
    *y = i;
    i++;
  }
}

void fernando_evil_1(int *a, int *b, int N) {
  int i;
  for (i = 0; i < N; i++) {
    a[i] = i;
    b[a[i]] = 0;
  }
  for (i = 0; i < N; i++) {
    int j;
    b[i] += 0;
    for (j = 0; j < N; j++) {
      b[a[j]] = i;
    }
  }
}

