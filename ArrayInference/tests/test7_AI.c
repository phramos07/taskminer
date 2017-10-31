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

