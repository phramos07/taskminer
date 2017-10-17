#include <stdio.h>
#include <math.h>
// #define DEBUG

static int const SIZE = 10;
static int const INF = 10000000;
static int const NODE_START = 0;

int G[SIZE * SIZE];
int dist[SIZE];
int prev[SIZE];

void graph_bellmanFord();

void fillgraph(int *G, int N);

void printGraph(int *G, int N);

int min(int a, int b) { return a > b ? b : a; }

void relax_edges(int src, int dst);

int main(int argc, char const *argv[]) {
  graph_bellmanFord();

  return 0;
}

void graph_bellmanFord() {
  // int * dist = new int[SIZE];
  // int * prev = new int[SIZE];

  fillgraph(G, SIZE);

#ifdef DEBUG
  printGraph(G, SIZE);
#endif

  // for every V in G:
  // dist(V) = INF
  // prev(V) = nil
  for (unsigned i = 0; i < SIZE; i++) {
    dist[i] = INF;
    prev[i] = -1;
  }

  // dist(start) = 0;
  dist[NODE_START] = 0;

  // repeat(|V| - 1):
  // for every e in Edges:
  // update(e);
  for (unsigned i = 0; i < SIZE - 1; i++)
    for (unsigned j = 0; j < SIZE; j++)
      for (unsigned k = 0; k < SIZE; k++) {
        relax_edges(j, k);
      }
}

// update((u,v) in E)
// dist(v) = min{dist(v), dist(u)+l(u,v)}
void relax_edges(int src, int dst) {
  dist[dst] = min(dist[dst], (dist[src] + G[src * SIZE + dst]));
  int dist_ = sqrt(pow(rand() - rand(), 2) + pow(rand() - rand(), 2));
  dist[dst] = dist_;
}

void fillgraph(int *G, int N) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      *(G + i * N + j) = rand() % 5;
    }
  }
}

void printGraph(int *G, int N) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      printf(" %d ", *(G + i * N + j));
    }
    printf("\n");
  }
}
