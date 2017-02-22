#include <iostream>
#include <omp.h>
#include <math.h>
#define N 5

void bfs(int** G, int* node, int* neigh);

void fillgraph(int** G);

void printGraph(int** G);

int main()
{
	int **G;
	G = (int**)malloc(N*sizeof(int*));
	for (int i=0; i<N; i++)
		G[i] = (int*)malloc(N*sizeof(int));
	int neigh[N];
	int visited[N];

	fillgraph(G);
	printGraph(G);

	// bfs(G, G[0], neigh);

	for (int i = 0; i < N; i++)
		std::cout << "Node " << i << " has " << neigh[i] << " neighbors" << std::endl;

	return 0;
}

void bfs(int** G, int* node,int* neigh)
{
	for (int i=0; i<N; i++)
		if (node[i])
		{
			neigh[i]++;
			bfs(G, G[i], neigh);			
		}
}

void fillgraph(int** G)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			G[i][j] = rand()%N;
		}
}

void printGraph(int** G)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			std::cout << G[i][j] << " ";
		}
		std::cout << std::endl;
	}
}


