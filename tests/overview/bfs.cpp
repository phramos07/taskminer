#include <iostream>
#include <vector>
#include <map>
#include <omp.h>
#include <math.h>
#include <chrono>
#include <thread>
#define N 10000
#define MAX_COORD 100
#define MAX_DIST 1000000.0;
// #define DEBUG

struct Coord
{
	int x;
	int y;
};

std::map<int, Coord> nodesCoord;
std::vector<double> nodesMinDist(N);
std::vector<int> nodesMinDistIndex(N);

void bfs(int* G, int* node, int index, bool* visited);

void fillgraph(int* G);

void printGraph(int* G);

void fillRandomTree(int* G);

void findNearestNeighbor(int src, int dst);

int main(int argc, char* argv[])
{
	int* G = new int[N*N];
	int* neigh = new int[N];
	bool* visited = new bool[N];
	for (unsigned i = 0; i<N; i++)
	{
		visited[i] = false;
		neigh[i] = 0;
		nodesMinDist[i] = MAX_DIST;
	}

	fillgraph(G);
	// fillRandomTree(G);

	bfs(G, &G[0], 0, visited);

	#ifdef DEBUG
		printGraph(G);

		for (unsigned i = 0; i < N; i++)
			std::cout << "Node " << i << " has " << neigh[i] << " in-edges" << std::endl;

		for (unsigned i = 0; i< N; i++)
		{
			std::cout << "Node "
								<< i
								<< " Min dist, node: "
								<< nodesMinDistIndex[i]
								<< " at "
								<< nodesMinDist[i]
								<< "\n";
		}		
	#endif

	delete G;
	delete neigh;
	delete visited;

	return 0;
}

void bfs(int* G, int* node, int index, bool* visited)
{
	if (!visited[index])
	{
		visited[index] = true;
		#pragma omp parallel
		#pragma omp single
		for (long unsigned i=0; i<N; i++)
			if (*(node + i) != 0)
			{
				findNearestNeighbor(index, i);
				#pragma omp task depend(in:G[i*N])
				bfs(G, &G[i*N], i, visited);			
			}
	}
	return;
}

void findNearestNeighbor(int src, int dst)
{
	double dist = sqrt(pow(nodesCoord[src].x - nodesCoord[dst].x, 2) + pow(nodesCoord[src].y - nodesCoord[dst].y, 2));
	if (dist < nodesMinDist[src])
	{
		nodesMinDist[src] = dist;
		nodesMinDistIndex[src] = dst;
	}
}

void fillgraph(int* G)
{
	for (long unsigned i = 0; i < N; i++)
	{
		for (long unsigned j = 0; j < N; j++)
		{
			*(G + i*N + j) = rand()%5;
		}
		nodesCoord[i].x = rand()%MAX_COORD;
		nodesCoord[i].y = rand()%MAX_COORD;		
	}
}

void fillRandomTree(int* G)
{
	int* pruferCode = new int[N-2];
	for (unsigned i = 0; i < N-2; i++)
		pruferCode[i] = rand()%100;

	int* degree = new int[N];
}

void printGraph(int* G)
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


