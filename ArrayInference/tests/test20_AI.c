
void func(int a) {
  int n[100];
  int i;
  long long int AI1[8];
  AI1[0] = a + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = 4 + AI1[1];
  AI1[3] = AI1[2] + 4;
  AI1[4] = AI1[3] / 4;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  AI1[7] = AI1[6] - 1;
  #pragma acc data pcopyin(n[1:AI1[7]]) pcopyout(n[1:AI1[7]])
  #pragma acc kernels
  #pragma acc loop independent 
  for (i = 0; i < a; i++) {
    n[i + 1] = n[i + 1] + 1;
  }
}

