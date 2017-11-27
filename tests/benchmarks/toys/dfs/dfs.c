#include <math.h>
#include <stdio.h>
#define N 5000
#define MAX_COORD 100
#define MAX_DIST 100000;
// #define DEBUG

struct Coord {
  int x;
  int y;
};

int nodesCoordX[N];
int nodesCoordY[N];
int nodesMinDist[N];
int nodesMinDistIndex[N];
int G[N * N];
int neigh[N];
int visited[N];

// std::map<int, Coord> nodesCoord;
// std::vector<double> nodesMinDist(N);
// std::vector<int> nodesMinDistIndex(N);

void dfs(int *G, int *node, int index);

void fillgraph(int *G);

void printGraph(int *G);

void findNearestNeighbor(int src, int dst);

int main(int argc, char *argv[]) {
  // int* G = new int[N*N];
  // int* neigh = new int[N];
  // bool* visited = new bool[N];
  for (unsigned i = 0; i < N; i++) {
    visited[i] = 0;
    neigh[i] = 0;
    nodesMinDist[i] = MAX_DIST;
  }

  fillgraph(G);
  dfs(G, &G[0], 0);

#ifdef DEBUG
  printGraph(G);

  for (unsigned i = 0; i < N; i++)
    printf("Node %d has %d in-edges\n", i, neigh[i]);

// for (unsigned i = 0; i< N; i++)
// {
// 	std::cout << "Node "
// 						<< i
// 						<< " Min dist, node: "
// 						<< nodesMinDistIndex[i]
// 						<< " at "
// 						<< nodesMinDist[i]
// 						<< "\n";
// }
#endif

  // delete [] G;
  // delete [] neigh;
  // delete [] visited;

  return 0;
}

void dfs(int *G, int *node, int index) {
  if (!visited[index]) {
    visited[index] = 1;
    for (unsigned i = 0; i < N; i++)
      if (*(node + i) != 0) {
        // recursive call
        dfs(G, &G[i * N], i);

        // eventual computations
        neigh[i]++;
        double dist = sqrt(pow(nodesCoordX[index] - nodesCoordX[i], 2) +
                           pow(nodesCoordY[index] - nodesCoordY[i], 2));
        if (dist < nodesMinDist[index]) {
          nodesMinDist[index] = dist;
          nodesMinDistIndex[index] = i;
        }
      }
  }
  return;
}

void findNearestNeighbor(int src, int dst) {
  double dist = sqrt(pow(nodesCoordX[src] - nodesCoordX[dst], 2) +
                     pow(nodesCoordY[src] - nodesCoordY[dst], 2));
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
    nodesCoordX[i] = rand() % MAX_COORD;
    nodesCoordY[i] = rand() % MAX_COORD;
  }
}

void printGraph(int *G) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      printf(" %d ", *(G + i * N + j));
    }
    printf("\n");
  }
}

//