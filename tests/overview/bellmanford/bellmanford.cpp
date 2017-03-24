#include "../lib/Graph.hpp"
#include <cmath>
#include <ctime>
#include <omp.h>

static const int SIZE = 1500;

Graph G(SIZE);

void graph_bellmanFord();

void udpate();

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
	for (unsigned i = 0; i < G.size; i++)
		G[i]->visited = false;

	int * dist new int(SIZE);


	//for every V in G:
	// dist(V) = INF
	// prev(V) = nil

	//dist(start) = 0;

	//repeat(|V| - 1):
	//for every e in Edges:
	//update(e);
}

//update((u,v) in E)
//dist(v) = min{dist(v), dist(u)+l(u,v)}