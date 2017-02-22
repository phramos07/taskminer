#include <iostream>
#include <omp.h>
#include <math.h>
#define N 10000

void bfs(long unsigned* G, long unsigned* node, int* neigh, int index, bool* visited);

void fillgraph(long unsigned* G);

void printGraph(long unsigned* G);

int main(int argc, char* argv[])
{
	long unsigned* G = (long unsigned*)malloc(N*N*sizeof(long unsigned));
	int* neigh = (int*)malloc(N*sizeof(int));
	bool* visited = (bool*)malloc(N*sizeof(bool));
	for (unsigned i = 0; i<N; i++)
	{
		visited[i] = false;
		neigh[i] = 0;
	}

	fillgraph(G);
	// printGraph(G);

	bfs(G, &G[0], neigh, 0, visited);

	// for (int i = 0; i < N; i++)
	// 	std::cout << "Node " << i << " has " << neigh[i] << " in-edges" << std::endl;

	free(G);
	free(neigh);
	free(visited);

	return 0;
}

void bfs(long unsigned* G, long unsigned* node,int* neigh, int index, bool* visited)
{
	if (!visited[index])
	{
		visited[index] = true;
		#pragma omp parallel
		#pragma omp single
		for (long unsigned i=0; i<N; i++)
			if (*(node + i) != 0)
			{
				neigh[i]++;
				#pragma omp task depend(in:G[i*N])
				bfs(G, &G[i*N], neigh, i, visited);			
			}
	}
	return;
}

void fillgraph(long unsigned* G)
{
	for (long unsigned i = 0; i < N; i++)
		for (long unsigned j = 0; j < N; j++)
		{
			*(G + i*N + j) = rand()%4;
		}
}

void printGraph(long unsigned* G)
{
	for (long unsigned i = 0; i < N; i++)
	{
		for (long unsigned j = 0; j < N; j++)
		{
			std::cout << *(G + i*N + j) << " ";
		}
		std::cout << std::endl;
	}
}


