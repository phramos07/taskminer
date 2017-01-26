#include <stdio.h>
#include <stdlib.h>

void Tenta(int no_of_nodes) {
  // allocate mem for the result on host side
  int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes * no_of_nodes);
  int *h_cost_gpu = (int *)malloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++) {
    for (int j = 0; j < no_of_nodes; j++) {
      h_cost[j] = j;
    }
    h_cost_gpu[i] = i;
  }
  int sum1 = 0, sum2 = 0;
  long long int AI2[6];
  AI2[0] = no_of_nodes + -1;
  AI2[1] = 4 * AI2[0];
  AI2[2] = AI2[1] + 1;
  AI2[3] = AI2[2] / 4;
  AI2[4] = (AI2[3] > 0);
  AI2[5] = (AI2[4] ? AI2[3] : 0);
  char RST_AI2 = 0;
  RST_AI2 |= !((h_cost + 0 > h_cost_gpu + AI2[5])
  || (h_cost_gpu + 0 > h_cost + AI2[5]));
  #pragma acc data pcopy(h_cost[0:AI2[5]],h_cost_gpu[0:AI2[5]]) if(!RST_AI2)
  #pragma acc kernels if(!RST_AI2)
  for (int k = 0; k < no_of_nodes; k++) {
    sum1 = sum1 + h_cost[k];
    sum2 += h_cost_gpu[k];
  }
  printf("%d %d\n", sum1, sum2);
}

int main(int argc, char *argv[]) {
  int *ml = (int *)malloc(sizeof(int) * 130);
  Tenta(10000);
  return 0;
}

