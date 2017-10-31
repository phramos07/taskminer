#include <stdlib.h>
typedef struct var {
  long int a;
  long int b;
  long int c[100];
} var;

int main(int argc, char *argv[]) {
  int b = atoi(argv[1]);
  // int b = 10000;
  var *n = (var *)malloc(sizeof(var) * b);
  // int *n = (int*) malloc(sizeof(int) * b);
  long long int AI1[9];
  AI1[0] = b + -1;
  AI1[1] = 816 * AI1[0];
  AI1[2] = 8 + AI1[1];
  AI1[3] = AI1[2] > AI1[1];
  AI1[4] = (AI1[3] ? AI1[2] : AI1[1]);
  AI1[5] = AI1[4] + 1;
  AI1[6] = AI1[5] / 816;
  AI1[7] = (AI1[6] > 0);
  AI1[8] = (AI1[7] ? AI1[6] : 0);
  #pragma acc data pcopy(n[0:AI1[8]])
  {
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent 
  for (int i = 0; i < b; i++) {
    #pragma acc loop independent 
    for (int j = 0; j < b; j++)
      n[i].a = 0;
    n[i].b = i;
  }
}

  long long int AI2[8];
  AI2[0] = 2 + b;
  AI2[1] = 816 * AI2[0];
  AI2[2] = AI2[1] * 1;
  AI2[3] = AI2[2] + 1;
  AI2[4] = AI2[3] / 816;
  AI2[5] = (AI2[4] > 0);
  AI2[6] = (AI2[5] ? AI2[4] : 0);
  AI2[7] = AI2[6] - 2;
  #pragma acc data pcopy(n[2:AI2[7]])
  {
  #pragma acc kernels if(!RST_AI2)
  #pragma acc loop independent 
  for (int i = 0; i <= b; i++)
    n[i + 2].a = n[i + 2].b + 1;
}

  return 0;
}

