void func(int a, int b) {
  int n[100];
  long long int AI1[8];
  AI1[0] = b + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = 4 + AI1[1];
  AI1[3] = AI1[2] + 4;
  AI1[4] = AI1[3] / 4;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  AI1[7] = AI1[6] - 1;
  #pragma omp target data map(tofrom: n[1:AI1[7]])
  {
  #pragma omp target if(!RST_AI1)
  #pragma omp parallel for 
  for (int i = 0; i < b; i++) {
    n[i + 1] = n[i + 1] + 1;
  }
}
}

