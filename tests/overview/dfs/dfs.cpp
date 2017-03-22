#include "../lib/Graph.hpp"
#include <cmath>
#include <ctime>
#include <omp.h>

static const int SIZE = 1500;

Graph G(SIZE);

double dfs(Node<NT, ET>* N);

void graph_dfs();

int main(int argc, char const *argv[])
{
	clock_t begin = std::clock();

	graph_dfs();

	clock_t end = std::clock();

	double elapsed_time = double(end-begin) / CLOCKS_PER_SEC;

	std::cout << "\nElapsed time: " << elapsed_time << "s \n";

	return 0;
}

void graph_dfs()
{
	for (unsigned i = 0; i < G.size; i++)
		G[i]->visited = false;

	for (unsigned i = 0; i < G.size; i++)
		dfs(G[i]);
}

double dfs(Node<NT, ET>* N)
{
	double dist = 0.0;
	if (!N->visited)
	{
		N->visited = true;

		//Eventual computations
		dist += sqrt(pow(N->index + N->weight, 2) + pow(N->index + 4.0*N->weight, 2));

		#pragma omp parallel
		#pragma omp single
		for (unsigned i = 0; i < N->edges.size(); i++)
		{
			//dfs recursive visit
			int index = N->edges[i]->dst->index;
			#pragma omp task
			dfs(G[index]);
		}
	}

	return dist;
}

