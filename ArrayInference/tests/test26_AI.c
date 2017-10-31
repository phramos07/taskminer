void foo(int *v, int *s, int n, int m) {

  long long int AI1[25];
  AI1[0] = n + -1;
  AI1[1] = m * AI1[0];
  AI1[2] = m + -1;
  AI1[3] = AI1[1] + AI1[2];
  AI1[4] = AI1[3] * 4;
  AI1[5] = 4 * AI1[2];
  AI1[6] = 4 * AI1[0];
  AI1[7] = AI1[5] > AI1[6];
  AI1[8] = (AI1[7] ? AI1[5] : AI1[6]);
  AI1[9] = (long long int) AI1[8];
  AI1[10] = AI1[4] > AI1[9];
  AI1[11] = (AI1[10] ? AI1[4] : AI1[9]);
  AI1[12] = (long long int) AI1[11];
  AI1[13] = AI1[12] + 4;
  AI1[14] = AI1[13] / 4;
  AI1[15] = (AI1[14] > 0);
  AI1[16] = (AI1[15] ? AI1[14] : 0);
  AI1[17] = AI1[5] > AI1[8];
  AI1[18] = (AI1[17] ? AI1[5] : AI1[8]);
  AI1[19] = AI1[6] > AI1[18];
  AI1[20] = (AI1[19] ? AI1[6] : AI1[18]);
  AI1[21] = AI1[20] + 4;
  AI1[22] = AI1[21] / 4;
  AI1[23] = (AI1[22] > 0);
  AI1[24] = (AI1[23] ? AI1[22] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !((s + 0 > v + AI1[16])
  || (v + 0 > s + AI1[24]));
  #pragma acc data pcopy(s[0:AI1[24]],v[0:AI1[16]]) if(!RST_AI1)
  {
  #pragma acc kernels if(!RST_AI1)
  for (int i = 0; i < n; i++)
    v[i] = s[i];

  #pragma acc kernels if(!RST_AI1)
  for (int j = 0; j < m; j++)
    s[j] = v[j];

  #pragma acc kernels if(!RST_AI1)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      v[i * m + j] = s[j] + s[i];
}
}

void foo2(int n, int *vect) {

  long long int AI1[6];
  AI1[0] = n + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = AI1[1] + 4;
  AI1[3] = AI1[2] / 4;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  #pragma acc data pcopy(vect[0:AI1[5]])
  {
  #pragma acc kernels if(!RST_AI1)
  for (int i = 0; i < n; i++)
    vect[i] = vect[i] + 1;
}
}

