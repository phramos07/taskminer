#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "../../include/time_common.h"
// #define DEBUG

void graph_bellmanFord(int **G, int *dist, int *prev, int node_start, int size);

void fillgraph(int **G, int N);

void printGraph(int **G, int N);

int min(int a, int b) { return a > b ? b : a; }

void relax_edges(int **G, int src, int dst, int *dist);

int main(int argc, char const *argv[]) {
  Instance *I = newInstance(100);
  clock_t beg, end;

  int size = atoi(argv[1]);
  //create graph
  int **G = (int **)malloc(sizeof(int *) * size);
  int i;
  for (i = 0; i < size; i++)
    G[i] = (int *)malloc(sizeof(int) * size);

  int *dist = (int *)malloc(sizeof(int) * size);
  int *prev = (int *)malloc(sizeof(int) * size);
  memset(dist, 0, sizeof(int) * size);
  memset(prev, 0, sizeof(int) * size);

  fillgraph(G, size);
  beg = clock();
  graph_bellmanFord(G, dist, prev, 0, size);
  end = clock();
  addNewEntry(I, size, getTimeInSecs(end - beg));

  writeResultsToOutput(stdout, I);
  freeInstance(I);

  return 0;
}

void graph_bellmanFord(int **G, int *dist, int *prev, int node_start, int size) {
// int * dist = new int[SIZE];
// int * prev = new int[SIZE];

#ifdef DEBUG
  printGraph(G, size);
#endif

  //for every V in G:
  // dist(V) = INF
  // prev(V) = nil
  for (unsigned i = 0; i < size; i++) {
    dist[i] = INT_MAX;
    prev[i] = -1;
  }

  //dist(start) = 0;
  dist[node_start] = 0;

  //repeat(|V| - 1):
  //for every e in Edges:
  //update(e);
  #pragma omp parallel
  #pragma omp single
  for (unsigned i = 0; i < size - 1; i++)
    for (unsigned j = 0; j < size; j++)
      for (unsigned k = 0; k < size; k++) {
        cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
        #pragma omp task untied default(shared) depend(in:G) depend(inout:dist)
        relax_edges(G, j, k, dist);
      }
#pragma omp taskwait
}

//update((u,v) in E)
//dist(v) = min{dist(v), dist(u)+l(u,v)}
void relax_edges(int **G, int src, int dst, int *dist) {
  dist[dst] = min(dist[dst], (dist[src] + G[src][dst]));
  int dist_ = sqrt(pow(rand() - rand(), 2) + pow(rand() - rand(), 2));
  dist[dst] = dist_;
}

void fillgraph(int **G, int N) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      G[i][j] = rand() % 5;
    }
  }
}

void printGraph(int **G, int N) {
  for (long unsigned i = 0; i < N; i++) {
    for (long unsigned j = 0; j < N; j++) {
      printf(" %d ", G[i][j]);
    }
    printf("\n");
  }
}

