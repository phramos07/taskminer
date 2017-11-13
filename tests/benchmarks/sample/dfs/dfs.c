#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_COORD 100
#define MAX_DIST 100000;
// #define DEBUG

void dfs(int** G, int index, int* visited, int size);

void fillgraph(int* G) { return; };

void printGraph(int* G){ return; };

int** newGraph(int size) { int** v, i; v = (int**) malloc(sizeof(int*)*size); for (i = 0; i < size; i++) v[i] = (int*)malloc(sizeof(int)*size); return v; }

void euclidianDist(int src, int dst) { return; };

int main(int argc, char* argv[])
{
	int size = 500;
	int** G = newGraph(size);
	int* visited = malloc(sizeof(int)*size);

	dfs(G, 0, visited, size);

	return 0;
}

void dfs(int** G, int index, int* visited, int size)
{
	if (!visited[index])
	{
		visited[index] = 1;
		#pragma omp parallel
		#pragma omp single
		for (unsigned i=0; i<size; i++)
		{
			if (G[index][i] != 0)
			{
				#pragma omp task depend(in: G) shared(visited) untied
				dfs(G, i, visited, size);			
				euclidianDist(index, i);
			}			
		}
	}
	return;
}