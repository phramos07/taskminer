#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include <stdlib.h>
#include <stdio.h>
// #define SIZE 10000

void fillgraph(int *G, int N) {
  #pragma omp parallel
  #pragma omp single
  for (long unsigned i = 0; i < N; i++) {
    {
    long long int TM9[7];
    TM9[0] = 4 * N;
    TM9[1] = TM9[0] * N;
    TM9[2] = TM9[1] + TM9[0];
    TM9[3] = TM9[2] + 4;
    TM9[4] = TM9[3] / 4;
    TM9[5] = (TM9[4] > 0);
    TM9[6] = (TM9[5] ? TM9[4] : 0);
    int tmc2 = N * (19);
    int tm_cost1 = (10 + tmc2);
    #pragma omp task depend(inout: G[0:TM9[6]]) if(tm_cost1 > 6000)
    {
    for (long unsigned j = 0; j < N; j++) {
      *(G + i * N + j) = rand() % 50 + rand() % 50 + 3;
    }
  }
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

struct edge {
  int src, dst, weight;
};

typedef struct edge edgeType;

edgeType *edgeTab;
int numVertices;
int *parent, *weight, numTrees;
int *bestEdgeNum;
int *G;

int find(int x) {
  int i, j, root;
  for (i = x; parent[i] != i;)
    i = parent[i];

  root = i;

  /* path compression */
  for (i = x; parent[i] != i;) {
    j = parent[i];
    parent[i] = root;
    i = j;
  }

  return root;
}

void union_(int i, int j) {
  if (weight[i] > weight[j]) {
    parent[j] = i;
    weight[i] += weight[j];
  } else {
    parent[i] = j;
    weight[j] += weight[i];
  }
}

int main(int argc, char *argv[]) {
  int i, j, MSTweight = 0;
  int root1, root2;
  int usefulEdges;

  int SIZE = atoi(argv[1]);
  numVertices = SIZE;
  edgeTab = (edgeType *)malloc(SIZE * SIZE * sizeof(edgeType));
  parent = (int *)malloc(SIZE * sizeof(int));
  weight = (int *)malloc(SIZE * sizeof(int));
  bestEdgeNum = (int *)malloc(SIZE * sizeof(int));
  G = (int *)malloc(SIZE * SIZE * sizeof(int));

  fillgraph(G, SIZE);
  if (SIZE < 20)
    printGraph(G, SIZE);

  // fillEdgeInfo
  for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++) {
      edgeTab[i * SIZE + j].src = i;
      edgeTab[i * SIZE + j].dst = j;
      edgeTab[i * SIZE + j].weight =
          G[i * SIZE + j] != 0 ? G[i * SIZE + j] : -1;
    }

  for (i = 0; i < numVertices; i++) {
    parent[i] = i;
    weight[i] = 1;
  }

  numTrees = numVertices; // Each vertex is initially in its own subtree
  usefulEdges =
      SIZE * SIZE; // An edge is useful if the two vertices are separate

  #pragma omp parallel
  #pragma omp single
  while (numTrees > 1 && usefulEdges > 0) {
    for (i = 0; i < numVertices; i++)
      bestEdgeNum[i] = (-1);

    usefulEdges = 0;

    for (i = 0; i < SIZE * SIZE; i++) {
      if (edgeTab[i].weight == -1)
        continue;
      root1 = find(edgeTab[i].src);
      root2 = find(edgeTab[i].dst);
      if (root1 != root2) {
        usefulEdges++;
        if (bestEdgeNum[root1] == (-1) ||
            edgeTab[bestEdgeNum[root1]].weight > edgeTab[i].weight)
          bestEdgeNum[root1] = i; // Have a new best edge from this component

        if (bestEdgeNum[root2] == (-1) ||
            edgeTab[bestEdgeNum[root2]].weight > edgeTab[i].weight)
          bestEdgeNum[root2] = i; // Have a new best edge from this component
      }
    }

    for (i = 0; i < numVertices; i++) {
      if (bestEdgeNum[i] != (-1)) {
        root1 = find(edgeTab[bestEdgeNum[i]].src);
        root2 = find(edgeTab[bestEdgeNum[i]].dst);
        if (root1 == root2)
          continue; // This round has already connected these components.
        MSTweight += edgeTab[bestEdgeNum[i]].weight;
        // printf("%d %d %d included in MST\n", edgeTab[bestEdgeNum[i]].src,
        //        edgeTab[bestEdgeNum[i]].dst, edgeTab[bestEdgeNum[i]].weight);
        cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
        #pragma omp task untied default(shared)
        union_(root1, root2);
        numTrees--;
      }
    }
    #pragma omp taskwait
    // printf("numTrees is %d\n", numTrees);
  }

  if (numTrees != 1)
    printf("MST does not exist\n");

  printf("Sum of weights of spanning edges %d\n", MSTweight);
}
