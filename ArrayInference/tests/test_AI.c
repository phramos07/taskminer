void fdtd_2d(float *Y, float *X, float *H, int m, int n) {
  int i = 0;
  if (shouldComputY(i)) {
    long long int AI1[24];
    AI1[0] = n + -1;
    AI1[1] = 4 * AI1[0];
    AI1[2] = AI1[1] + 4;
    AI1[3] = AI1[2] / 4;
    AI1[4] = (AI1[3] > 0);
    AI1[5] = (AI1[4] ? AI1[3] : 0);
    AI1[6] = -1 * n;
    AI1[7] = AI1[6] * 4;
    AI1[8] = AI1[7] < 0;
    AI1[9] = (AI1[8] ? AI1[7] : 0);
    AI1[10] = AI1[9] / 4;
    AI1[11] = (AI1[10] > 0);
    AI1[12] = (AI1[11] ? AI1[10] : 0);
    AI1[13] = AI1[6] + AI1[0];
    AI1[14] = AI1[13] * 4;
    AI1[15] = (long long int) AI1[1];
    AI1[16] = AI1[14] > AI1[15];
    AI1[17] = (AI1[16] ? AI1[14] : AI1[15]);
    AI1[18] = (long long int) AI1[17];
    AI1[19] = AI1[18] + 4;
    AI1[20] = AI1[19] / 4;
    AI1[21] = (AI1[20] > 0);
    AI1[22] = (AI1[21] ? AI1[20] : 0);
    AI1[23] = AI1[22] - AI1[12];
    char RST_AI1 = 0;
    RST_AI1 |= !((H + AI1[12] > Y + AI1[5])
    || (Y + 0 > H + AI1[23]));
    #pragma acc data pcopyin(H[AI1[12]:AI1[23]]) pcopy(Y[0:AI1[5]]) if(!RST_AI1)
    {
    #pragma acc kernels if(!RST_AI1)
    for (int j = 0; j < n; j++)
      Y[i * n + j] = Y[i * n + j] - 0.5 * (H[i * n + j] - H[(i - 1) * n + j]);
  }
  }

  long long int AI2[14];
  AI2[0] = n + -2;
  AI2[1] = 4 * AI2[0];
  AI2[2] = 4 + AI2[1];
  AI2[3] = AI2[2] + 4;
  AI2[4] = AI2[3] / 4;
  AI2[5] = (AI2[4] > 0);
  AI2[6] = (AI2[5] ? AI2[4] : 0);
  AI2[7] = AI2[6] - 1;
  AI2[8] = AI2[1] > AI2[2];
  AI2[9] = (AI2[8] ? AI2[1] : AI2[2]);
  AI2[10] = AI2[9] + 4;
  AI2[11] = AI2[10] / 4;
  AI2[12] = (AI2[11] > 0);
  AI2[13] = (AI2[12] ? AI2[11] : 0);
  char RST_AI2 = 0;
  RST_AI2 |= !((H + 0 > X + AI2[7])
  || (X + 1 > H + AI2[13]));
  #pragma acc data pcopyin(H[0:AI2[13]]) pcopy(X[1:AI2[7]]) if(!RST_AI2)
  {
  #pragma acc kernels if(!RST_AI2)
  for (int j = 1; j < n; j++)
    X[i * n + j] = X[i * n + j] - 0.5 * (H[i * n + j] - H[i * n + (j - 1)]);
}
}

