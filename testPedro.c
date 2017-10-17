#include <stdio.h>
#include <stdlib.h>

void bfs(int *G, int *node, int index, bool *visited) {
  if (!visited[index]) {
    visited[index] = true;
    for (unsigned i = 0; i < N; i++)
      if (*(node + i) != 0) {
        findNearestNeighbor(index, i);
        bfs(G, &G[i * N], i, visited);
      }
  }
}
