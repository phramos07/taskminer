#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define N 1000
#define MAX_COORD 100
#define MAX_DIST 1000000.0;
// #define DEBUG

struct Coord {
  int x;
  int y;
};

std::map<int, Coord> nodesCoord;
std::vector<double> nodesMinDist(N);
std::vector<int> nodesMinDistIndex(N);

void bfs(int *G, int *node, int index, bool *visited);

void fillgraph(int *G);

void printGraph(int *G);

void findNearestNeighbor(int src, int dst);

int main(int argc, char *argv[]) {
  int *G = new int[N * N];
  int *neigh = new int[N];
  bool *visited = new bool[N];
  for (unsigned i = 0; i < N; i++) {
    visited[i] = false;
    neigh[i] = 0;
    nodesMinDist[i] = MAX_DIST;
  }

  fillgraph(G);
  bfs(G, &G[0], 0, visited);

#ifdef DEBUG
  printGraph(G);

  for (unsigned i = 0; i < N; i++)
    std::cout << "Node " << i << " has " << neigh[i] << " in-edges"
              << std::endl;

  for (unsigned i = 0; i < N; i++) {
    std::cout << "Node " << i << " Min dist, node: " << nodesMinDistIndex[i]
              << " at " << nodesMinDist[i] << "\n";
  }
#endif

  delete[] G;
  delete[] neigh;
  delete[] visited;

  return 0;
}

void bfs(int *G, int *node, int index, bool *visited) {
  if (!visited[index]) {
    visited[index] = true;
    #pragma omp parallel
    #pragma omp single
    for (unsigned i = 0; i < N; i++)
      if (*(node + i) != 0) {
        // eventual computations
        double dist = sqrt(pow(nodesCoord[index].x - nodesCoord[i].x, 2) +
                           pow(nodesCoord[index].y - nodesCoord[i].y, 2));

        // recursive call
        long long int TM14[3];
        TM14[0] = i * 1000;
        TM14[1] = TM14[0] * 4;
        TM14[2] = (TM14[1] / 4);
        #pragma omp task depend(in:G,G[TM14[2]]) depend(inout:visited)
        bfs(G, &G[i * N], i, visited);
      }
  }
  return;
}

void findNearestNeighbor(int src, int dst) {
  double dist = sqrt(pow(nodesCoord[src].x - nodesCoord[dst].x, 2) +
                     pow(nodesCoord[src].y - nodesCoord[dst].y, 2));
  if (dist < nodesMinDist[src]) {
    nodesMinDist[src] = dist;
    nodesMinDistIndex[src] = dst;
  }
}

void fillgraph(int *G) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      *(G + i * N + j) = rand() % 5;
    }
    nodesCoord[i].x = rand() % MAX_COORD;
    nodesCoord[i].y = rand() % MAX_COORD;
  }
}

void printGraph(int *G) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      std::cout << *(G + i * N + j) << " ";
    }
    std::cout << std::endl;
  }
}

