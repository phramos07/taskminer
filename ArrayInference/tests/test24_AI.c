int *v;

void foo(int n) {
  int i = 0;
  long long int AI1[7];
  AI1[0] = n + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = AI1[1] + 4;
  AI1[3] = AI1[2] / 4;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  AI1[6] = AI1[5] - 0;
  #pragma acc data pcopy(v[0:AI1[6]])
  {
  #pragma acc kernels
  for (i = 0; i < n; i++)
    v[i] = v[i] + i;
}
}

