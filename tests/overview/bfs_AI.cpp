#include <iostream>
#include <stdlib.h>
#include <stdio.h>
// include <omp.h>
#define N 100

void bfs(int *G, int *node, int *neigh);

int main() {
  int *G = (int *)malloc(N * N * sizeof(int));
  int neigh[N];

  bfs(G, &G[0], neigh);

  return 0;
}

void bfs(int *G, int *node, int *neigh) {
	#pragma omp parallel
	#pragma omp single
  for (int i = 0; i < N; i++)
    if (node[i]) {
      neigh[i]++;
      long long int TM5[3];
      TM5[0] = i * 100;
      TM5[1] = TM5[0] * 4;
      TM5[2] = (TM5[1] / 4);
      #pragma omp task depend(inout:G,G[TM5[2]],neigh)
      bfs(G, &G[i * N], neigh);
    }
}

