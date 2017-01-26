void foo(int *a, int *b, int H, int n, int m) {
  long long int AI1[30];
  AI1[0] = -1 * H;
  AI1[1] = AI1[0] * 4;
  AI1[2] = 0 < AI1[1];
  AI1[3] = (AI1[2] ? 0 : AI1[1]);
  AI1[4] = AI1[3] / 4;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  AI1[7] = n + -1;
  AI1[8] = H * AI1[7];
  AI1[9] = m + -1;
  AI1[10] = AI1[8] + AI1[9];
  AI1[11] = AI1[10] * 4;
  AI1[12] = AI1[0] + AI1[8];
  AI1[13] = AI1[12] + AI1[9];
  AI1[14] = AI1[13] * 4;
  AI1[15] = AI1[11] > AI1[14];
  AI1[16] = (AI1[15] ? AI1[11] : AI1[14]);
  AI1[17] = (long long int) AI1[16];
  AI1[18] = AI1[17] + 4;
  AI1[19] = AI1[18] / 4;
  AI1[20] = (AI1[19] > 0);
  AI1[21] = (AI1[20] ? AI1[19] : 0);
  AI1[22] = AI1[21] - AI1[6];
  AI1[23] = -1 + AI1[8];
  AI1[24] = AI1[23] + AI1[9];
  AI1[25] = AI1[24] * 4;
  AI1[26] = AI1[25] + 4;
  AI1[27] = AI1[26] / 4;
  AI1[28] = (AI1[27] > 0);
  AI1[29] = (AI1[28] ? AI1[27] : 0);
  char RST_AI1 = 0;
  RST_AI1 |= !((a + AI1[6] > b + AI1[29])
  || (b + 0 > a + AI1[22]));
  #pragma acc data pcopyin(b[0:AI1[29]]) pcopy(a[AI1[6]:AI1[22]]) if(!RST_AI1)
  {
  #pragma acc kernels if(!RST_AI1)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      a[i * H + j] = a[(i - 1) * H + j] + b[i * H + (j - 1)];
}
}

