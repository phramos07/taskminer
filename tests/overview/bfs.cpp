#include <iostream>
#include <omp.h>
#define N 100

void bfs(int* G, int* node, int* neigh);

int main()
{
	int* G = (int*)malloc(N*N*sizeof(int));
	int neigh[N];

	bfs(G, &G[0], neigh);

	return 0;
}

void bfs(int* G, int* node, int* neigh)
{
	for (int i=0; i<N; i++)
		if (node[i])
		{
			neigh[i]++;
			bfs(G, &G[i*N], neigh);			
		}
}


