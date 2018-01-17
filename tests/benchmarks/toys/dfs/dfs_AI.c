#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// #define N 5000
#define MAX_COORD 100
#define MAX_DIST 100000;
// #define DEBUG

struct Coord
{
	int x;
	int y;
};

void dfs(int* G, int* node, int index, int* visited, int* neigh, int* nodesCoordX, int* nodesCoordY, int* nodesMinDist, int* nodesMinDistIndex, int N);

void fillgraph(int* G, int* nodesCoordX, int* nodesCoordY, int N);

void printGraph(int* G, int N);

void findNearestNeighbor(int src, int dst, int* nodesCoordX, int* nodesCoordY, int* nodesMinDist, int* nodesMinDistIndex);

int main(int argc, char* argv[])
{
	int N = atoi(argv[1]);

	int* nodesCoordX = (int*) malloc(sizeof(int)*N);
	int* nodesCoordY = (int*) malloc(sizeof(int)*N);
	int* nodesMinDist = (int*) malloc(sizeof(int)*N);
	int* nodesMinDistIndex = (int*) malloc(sizeof(int)*N);
	int* G = (int*) malloc(sizeof(int)*N*N);
	int* neigh = (int*) malloc(sizeof(int)*N);
	int* visited = (int*) malloc(sizeof(int)*N);

	// int* G = new int[N*N];
	// int* neigh = new int[N];
	// bool* visited = new bool[N];
	for (unsigned i = 0; i<N; i++)
	{
		visited[i] = 0;
		neigh[i] = 0;
		nodesMinDist[i] = MAX_DIST;
	}

	fillgraph(G, nodesCoordX, nodesCoordY, N);
	dfs(G, &G[0], 0, visited, neigh, nodesCoordX, nodesCoordY, nodesMinDist, nodesMinDistIndex, N);

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

	free(nodesCoordX);
	free(nodesCoordY);
	free(nodesMinDist);
	free(nodesMinDistIndex);
	free(G);
	free(neigh);
	free(visited);

	return 0;
}

void dfs(int* G, int* node, int index, int* visited, int* neigh, int* nodesCoordX, int* nodesCoordY, int* nodesMinDist, int* nodesMinDistIndex, int N)
{
	if (!visited[index])
	{
		visited[index] = 1;
		#pragma omp parallel
		#pragma omp single
		for (unsigned i=0; i<N; i++)
		{
			if (*(node + i) != 0)
			{
				//recursive call
				#pragma omp task depend(in:G, G[i])
				dfs(G, &G[i*N], i, visited, neigh, nodesCoordX, nodesCoordY, nodesMinDist, nodesMinDistIndex, N);			
				
				//eventual computations
				neigh[i]++; 
				double dist = sqrt(pow(nodesCoordX[index] - nodesCoordX[i], 2) + pow(nodesCoordY[index] - nodesCoordY[i], 2));
				if (dist < nodesMinDist[index])
				{
					nodesMinDist[index] = dist;
					nodesMinDistIndex[index] = i;
				}			
			}
		}
	}
	return;
}

void findNearestNeighbor(int src, int dst, int* nodesCoordX, int* nodesCoordY, int* nodesMinDist, int* nodesMinDistIndex)
{
	double dist = sqrt(pow(nodesCoordX[src] - nodesCoordX[dst], 2) + pow(nodesCoordY[src] - nodesCoordY[dst], 2));
	if (dist < nodesMinDist[src])
	{
		nodesMinDist[src] = dist;
		nodesMinDistIndex[src] = dst;
	}
}

void fillgraph(int* G, int* nodesCoordX, int* nodesCoordY, int N)
{
	for (long unsigned i = 0; i < N; i++)
	{
		for (long unsigned j = 0; j < N; j++)
		{
			*(G + i*N + j) = rand()%5;
		}
		nodesCoordX[i] = rand()%MAX_COORD;
		nodesCoordY[i] = rand()%MAX_COORD;		
	}
}

void printGraph(int* G, int N)
{
	for (long unsigned i = 0; i < N; i++)
	{
		for (long unsigned j = 0; j < N; j++)
		{
			printf(" %d ", *(G + i*N + j));
		}
		printf("\n");
	}
}


// 