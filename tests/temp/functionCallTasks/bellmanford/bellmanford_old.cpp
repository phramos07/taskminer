#include "../lib/Graph.hpp"
#include <cmath>
#include <ctime>
#include <omp.h>

static int const SIZE = 1000;
static int const INF = 10000000;
static int const NODE_START = 0;

Graph G(SIZE);

int graph[SIZE][SIZE];

void graph_bellmanFord();

void relax_edges(Edge<NT, ET>& E, int*& dist, int*& prev);

int main(int argc, char const *argv[])
{
	clock_t begin = std::clock();

	graph_bellmanFord();

	clock_t end = std::clock();

	double elapsed_time = double(end-begin) / CLOCKS_PER_SEC;

	std::cout << "\nElapsed time: " << elapsed_time << "s \n";

	return 0;
}

void graph_bellmanFord()
{
	int * dist = new int[SIZE];
	int * prev = new int[SIZE];

	//for every V in G:
	// dist(V) = INF
	// prev(V) = nil
	for (unsigned i = 0; i < SIZE; i++)
	{
		dist[i] = INF;
		prev[i] = -1;		
	}

	//dist(start) = 0;	
	dist[NODE_START] = 0;

	//repeat(|V| - 1):
	//for every e in Edges:
	//update(e);
	for (unsigned i = 0; i < SIZE - 1; i++)
		for (unsigned j = 0; j < SIZE; j++)
			for (unsigned k = 0; k < SIZE; k++)
			{

				dist[k] = std::min(dist[k], (dist[j] + graph[j][k]))
			}


}

//update((u,v) in E)
//dist(v) = min{dist(v), dist(u)+l(u,v)}
void relax_edges(int src, int dst, int* dist, int* prev)
{
	dist[dst] = std::min(dist[dst], (dist[src] + graph[src][dst]));	
}

